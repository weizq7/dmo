// One independently owned executor per background depth.
#include "exit_control.cu"

struct DepthState {
  int *published;       // Last token whose foreground has exited.
  int *error;           // Shared failure flag; never a job-selection lock.
  int *completed;       // Number of model loops completed for each released token.
  float *continuation;  // Last completed hidden state for each released token.
};

static DepthState depth_state(const uint64_t *v) {
  return {(int*)v[0], (int*)v[1], (int*)v[2], (float*)v[3]};
}
__device__ int sd_load(int *p) {
  return cuda::atomic_ref<int,cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}
__device__ void sd_store(int *p,int value) {
  cuda::atomic_ref<int,cuda::thread_scope_device>(*p).store(value,cuda::memory_order_release);
}

// Publication happens after the foreground's last loop and before vocabulary
// readout. A release store protects both hidden state and completed-loop count.
__global__ void sd_release(P p,DepthState s) {
  int pos=position(p);
  for(int i=threadIdx.x;i<p.width;i+=blockDim.x)
    s.continuation[pos*p.width+i]=p.history[pos*p.width+i];
  __syncthreads();
  if(threadIdx.x==0) {
    sd_store(s.completed+pos,int(p.depth_history[pos]));
    sd_store(s.published,pos);
  }
}

// There is exactly one executor per depth. Its own ready frontier identifies
// the sole next token it can process. No global task scan, ownership or lock.
// The published frontier is re-read, not fixed when this invocation starts.
__global__ void sd_select(P p,DepthState s,int depth,cudaGraphConditionalHandle run) {
  const auto began=now_ns();
  bool found=false;
  while(!sd_load(s.error)) {
    int published=sd_load(s.published);
    int pos=sd_load(p.ready+depth)+1;
    if(pos>published)break;
    if(pos<0||pos>=p.capacity){sd_store(s.error,21);break;}
    int done=sd_load(s.completed+pos);
    if(done==depth && sd_load(p.ready+depth-1)>=pos) {
      p.input[1]=pos;found=true;break;
    }
    // Foreground may have advanced this depth after the frontier snapshot.
    // Re-reading the frontier on the next iteration skips that completed work.
    if(now_ns()-began>10000000000ULL){sd_store(s.error,22);break;}
    __nanosleep(100);
  }
  cudaGraphSetConditional(run,found);
}

__global__ void sd_begin(P p,DepthState s,int depth) {
  int pos=position(p);
  if(threadIdx.x==0)p.trace[((pos*p.steps+depth)*2+1)*2]=now_ns();
  for(int i=threadIdx.x;i<p.width;i+=blockDim.x)
    p.hidden[i]=s.continuation[pos*p.width+i];
}

__global__ void sd_finish(P p,DepthState s,int depth) {
  int pos=position(p);
  for(int i=threadIdx.x;i<p.width;i+=blockDim.x)
    s.continuation[pos*p.width+i]=p.hidden[i];
  __syncthreads();
  if(threadIdx.x==0) {
    atomicAdd(p.stats+1,1ULL);
    p.trace[((pos*p.steps+depth)*2+1)*2+1]=now_ns();
    publish(p,depth);
    sd_store(s.completed+pos,depth+1);
  }
}

extern "C" uint64_t sd_make_foreground(int steps,int width,int capacity,
 const uint64_t *ptrs,const uint64_t *state,const uint64_t *loops,const uint64_t *gates,
 uint64_t embed,uint64_t readout) {
  Graph *g=new Graph;g->p=params(1,steps,width,capacity,ptrs);P p=g->p;
  DepthState s=depth_state(state);
  try {
    check(cudaGraphCreate(&g->graph,0));
    std::vector<cudaGraphConditionalHandle> enabled(steps);
    for(int d=0;d<steps;d++)
      check(cudaGraphConditionalHandleCreate(&enabled[d],g->graph,d==0,cudaGraphCondAssignDefault));
    void *begin[]={&p};auto prev=kernel(g->graph,nullptr,(void*)begin_fg,begin);
    prev=child(g->graph,prev,embed);
    for(int d=0;d<steps;d++) {
      cudaGraph_t branch;auto entry=conditional(g->graph,prev,enabled[d],branch);
      int background=0;void *before[]={&p,&d,&background};
      auto node=kernel(branch,nullptr,(void*)before_loop,before);
      node=child(branch,node,loops[d]);
      const float *gate=(const float*)gates[d];auto next=d+1<steps?enabled[d+1]:0;
      void *after[]={&p,&d,&gate,&next};kernel(branch,node,(void*)after_fg,after,256);
      prev=entry;
    }
    void *release[]={&p,&s};prev=kernel(g->graph,prev,(void*)sd_release,release,256);
    prev=child(g->graph,prev,readout);kernel(g->graph,prev,(void*)pack_output,begin);
    check(cudaGraphInstantiate(&g->exec,g->graph,0));return (uint64_t)g;
  }catch(std::exception &e){last_error=e.what();delete g;return 0;}
}

extern "C" uint64_t sd_make_worker(int steps,int width,int capacity,int depth,
 const uint64_t *ptrs,const uint64_t *state,uint64_t model_body) {
  Graph *g=new Graph;g->p=params(1,steps,width,capacity,ptrs);P p=g->p;
  DepthState s=depth_state(state);
  try {
    if(depth<1||depth>=steps||!model_body)throw std::runtime_error("Invalid fixed depth");
    check(cudaGraphCreate(&g->graph,0));cudaGraphConditionalHandle more;
    check(cudaGraphConditionalHandleCreate(&more,g->graph,0,cudaGraphCondAssignDefault));
    void *select[]={&p,&s,&depth,&more};auto first=kernel(g->graph,nullptr,(void*)sd_select,select);
    cudaGraphNodeParams loop{};loop.type=cudaGraphNodeTypeConditional;
    loop.conditional.handle=more;loop.conditional.type=cudaGraphCondTypeWhile;loop.conditional.size=1;
    cudaGraphNode_t node;check(cudaGraphAddNode(&node,g->graph,&first,1,&loop));
    auto body=loop.conditional.phGraph_out[0];void *work[]={&p,&s,&depth};
    auto step=kernel(body,nullptr,(void*)sd_begin,work,256);
    step=child(body,step,model_body);step=kernel(body,step,(void*)sd_finish,work,256);
    kernel(body,step,(void*)sd_select,select);
    check(cudaGraphInstantiate(&g->exec,g->graph,0));return (uint64_t)g;
  }catch(std::exception &e){last_error=e.what();delete g;return 0;}
}

extern "C" uint64_t sd_create_stream() {
  try {cudaStream_t stream;check(cudaStreamCreateWithPriority(&stream,cudaStreamNonBlocking,0));
    return (uint64_t)stream;
  }catch(std::exception &e){last_error=e.what();return 0;}
}
extern "C" int sd_destroy_stream(uint64_t stream) {
  try {check(cudaStreamDestroy((cudaStream_t)stream));return 0;}
  catch(std::exception &e){last_error=e.what();return 1;}
}
