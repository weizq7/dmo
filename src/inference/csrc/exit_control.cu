// GPU-controlled exit graphs. No host decision is made between recurrent steps.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

static thread_local std::string last_error;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct P {
  int mode, steps, width, capacity;
  int64_t *input;             // token, position
  double *uniforms, *survival;
  int64_t *control;           // selected depth, error
  int *ready;                // latest completed position at each depth
  unsigned long long *stats; // foreground loops, background loops, copied depths
  float *hidden, *selected, *history;
  int64_t *depth_history, *output, *word;
  unsigned long long *trace; // [position, depth, fg/bg, start/end]
};
static P params(int mode, int steps, int width, int capacity, const uint64_t *v) {
  return P{mode,steps,width,capacity,(int64_t*)v[0],(double*)v[1],(double*)v[2],
    (int64_t*)v[3],(int*)v[4],(unsigned long long*)v[5],(float*)v[6],
    (float*)v[7],(float*)v[8],(int64_t*)v[9],(int64_t*)v[10],(int64_t*)v[11],
    (unsigned long long*)v[12]};
}
__device__ unsigned long long now_ns() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ int position(P p) { return int(p.input[1]); }
__device__ void publish(P p, int depth) {
  cuda::atomic_ref<int,cuda::thread_scope_device> flag(p.ready[depth]);
  flag.store(position(p),cuda::memory_order_release);
}
__global__ void begin_fg(P p) {
  *p.survival=1.;p.control[0]=0;p.control[1]=0;
}
__global__ void before_loop(P p,int depth,int background) {
  if(p.mode==1 && !background) {
    cuda::atomic_ref<int,cuda::thread_scope_device> flag(p.ready[depth]);
    auto start=now_ns();
    while(flag.load(cuda::memory_order_acquire)<position(p)-1) {
      if(now_ns()-start>10000000000ULL) {p.control[1]=3;break;}
      __nanosleep(100);
    }
  }
  p.trace[((position(p)*p.steps+depth)*2+background)*2]=now_ns();
}
__global__ void after_fg(P p,int depth,const float *hazard,cudaGraphConditionalHandle next) {
  __shared__ bool chosen;
  if(threadIdx.x==0) {
    double lam=depth==p.steps-1 ? 1. : double(*hazard);
    if(!isfinite(lam)||lam<0.||lam>1.) p.control[1]=1;
    if(depth<p.steps-1) *p.survival *= 1.-lam;
    chosen=p.control[0]==0 && (depth==p.steps-1 || p.uniforms[0]<1.-*p.survival);
    if(chosen) {p.control[0]=depth+1;p.depth_history[position(p)]=depth+1;}
    if(next) cudaGraphSetConditional(next,p.control[0]==0 && p.control[1]==0);
    atomicAdd(p.stats,1ULL);
    p.trace[((position(p)*p.steps+depth)*2)*2+1]=now_ns();
    publish(p,depth);
  }
  __syncthreads();
  if(chosen) for(int i=threadIdx.x;i<p.width;i+=blockDim.x) {
    p.selected[i]=p.hidden[i];p.history[position(p)*p.width+i]=p.hidden[i];
  }
}
__global__ void pack_output(P p) {
  p.output[0]=*p.word;p.output[1]=p.control[0];p.output[2]=p.control[1];
  if(*p.word<0) p.output[2]=2;
}
__global__ void copy_kv(P p,uint16_t *keys,uint16_t *values,int layers,int heads,int dim) {
  int ex=int(p.control[0])-1,pos=position(p);
  // Storage: [depth, layer, head, position, head_dim]. B is exactly one.
  long long per_depth=(long long)layers*heads*dim;
  long long n=(long long)p.steps*per_depth;
  for(long long x=(long long)blockIdx.x*blockDim.x+threadIdx.x;x<n;x+=(long long)blockDim.x*gridDim.x) {
    int d=int(x/per_depth);if(d<=ex) continue;
    long long ld=x%per_depth,lh=ld/dim,offset=ld%dim;
    long long dst=((d*(long long)layers*heads+lh)*p.capacity+pos)*dim+offset;
    long long src=((ex*(long long)layers*heads+lh)*p.capacity+pos)*dim+offset;
    keys[dst]=keys[src];values[dst]=values[src];
  }
}
__global__ void publish_copies(P p) {
  for(int d=int(p.control[0]);d<p.steps;d++) publish(p,d);
  atomicAdd(p.stats+2,(unsigned long long)(p.steps-p.control[0]));
}

struct Graph { cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;P p;
  ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
static cudaGraphNode_t kernel(cudaGraph_t graph,cudaGraphNode_t prev,void *fn,void **args,int threads=1) {
  cudaKernelNodeParams k{};k.func=fn;k.gridDim=dim3(1);k.blockDim=dim3(threads);k.kernelParams=args;
  cudaGraphNode_t node;check(cudaGraphAddKernelNode(&node,graph,prev?&prev:nullptr,prev?1:0,&k));return node;
}
static cudaGraphNode_t child(cudaGraph_t graph,cudaGraphNode_t prev,uint64_t raw) {
  cudaGraphNode_t node;check(cudaGraphAddChildGraphNode(&node,graph,prev?&prev:nullptr,prev?1:0,(cudaGraph_t)raw));return node;
}
static cudaGraphNode_t conditional(cudaGraph_t graph,cudaGraphNode_t prev,cudaGraphConditionalHandle h,cudaGraph_t &body) {
  cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=h;
  p.conditional.type=cudaGraphCondTypeIf;p.conditional.size=1;
  cudaGraphNode_t node;check(cudaGraphAddNode(&node,graph,prev?&prev:nullptr,prev?1:0,&p));
  body=p.conditional.phGraph_out[0];return node;
}
extern "C" const char *ce_error(){return last_error.c_str();}
extern "C" uint64_t ce_make_fg(int mode,int steps,int width,int capacity,
  const uint64_t *ptrs,const uint64_t *loops,const uint64_t *gates,uint64_t embed,uint64_t readout) {
  Graph *g=new Graph;g->p=params(mode,steps,width,capacity,ptrs);P p=g->p;
  try {
    check(cudaGraphCreate(&g->graph,0));
    std::vector<cudaGraphConditionalHandle> handles(steps,0);
    if(mode)for(int d=0;d<steps;d++)check(cudaGraphConditionalHandleCreate(&handles[d],g->graph,d==0,cudaGraphCondAssignDefault));
    void *a[]={&p};auto prev=kernel(g->graph,nullptr,(void*)begin_fg,a);
    prev=child(g->graph,prev,embed);
    for(int d=0;d<steps;d++) {
      cudaGraph_t body=g->graph;cudaGraphNode_t root=nullptr,body_prev=prev;
      if(mode){root=conditional(g->graph,prev,handles[d],body);body_prev=nullptr;}
      int bg=0;void *b[]={&p,&d,&bg};body_prev=kernel(body,body_prev,(void*)before_loop,b);
      body_prev=child(body,body_prev,loops[d]);
      const float *gate=(const float*)gates[d];auto next=d+1<steps?handles[d+1]:0;
      void *c[]={&p,&d,&gate,&next};body_prev=kernel(body,body_prev,(void*)after_fg,c,256);
      prev=mode?root:body_prev;
    }
    prev=child(g->graph,prev,readout);kernel(g->graph,prev,(void*)pack_output,a);
    check(cudaGraphInstantiate(&g->exec,g->graph,0));return (uint64_t)g;
  } catch(std::exception &e) {last_error=e.what();delete g;return 0;}
}
extern "C" int ce_launch(uint64_t graph,uint64_t stream) {
  try{check(cudaGraphLaunch(((Graph*)graph)->exec,(cudaStream_t)stream));return 0;}
  catch(std::exception &e){last_error=e.what();return 1;}
}
extern "C" int ce_copy_kv(uint64_t graph,uint64_t stream,uint64_t keys,uint64_t values,int layers,int heads,int dim) {
  try{auto *g=(Graph*)graph;copy_kv<<<128,256,0,(cudaStream_t)stream>>>(g->p,(uint16_t*)keys,(uint16_t*)values,layers,heads,dim);
    publish_copies<<<1,1,0,(cudaStream_t)stream>>>(g->p);check(cudaGetLastError());return 0;}
  catch(std::exception &e){last_error=e.what();return 1;}
}
extern "C" void ce_destroy(uint64_t graph){delete (Graph*)graph;}
