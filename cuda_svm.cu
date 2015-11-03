#include <stdio.h>
#include <cuda_runtime.h>

#include "svm.h"
#include "cuda_svm.h"

typedef signed char schar;

template <class T> __device__ static inline T min(T x,T y) { return (x<y)?x:y; }

template <class T> __device__ static inline T max(T x,T y) { return (x>y)?x:y; }

template <class T> __device__ static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }

template <class S, class T> __device__ static inline void clone(T*& dst, S* src, int n)
{
    dst = (T *)malloc(sizeof(T) * n);
    memcpy((void *)dst, (void *)src, sizeof(T)*n);
}

__device__ static inline float powi(float base, int times)
{
    float tmp = base, ret = 1.0;

    for(int t=times; t>0; t/=2)
    {
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}



#define INF HUGE_VAL
#define TAU 1e-12



__device__ struct svm_model *cuda_device_svm_train_no_prob(const struct svm_problem *prob, const struct svm_parameter *param);
__device__ float cuda_svm_predict_values(const svm_model *model, const svm_node *x, float* dec_values);
__device__ void cuda_svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, float *target);






class CUDA_Rand
{
private:
    int seed;
public:
    __device__ CUDA_Rand()
    {
        seed = 0;
    }
    
    __device__ int rand_int(const int max)
    {
        seed = ((seed * 1103515245) + 12345) & 0x7fffffff;
        return seed%max;
    }
};


//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class CUDA_Cache
{
public:
    __device__ CUDA_Cache(int l,long int size);
    __device__ ~CUDA_Cache();

    // request data [0,len)
    // return some position p where [p,len) need to be filled
    // (p >= len if nothing needs to be filled)
    __device__ int get_data(const int index, float **data, int len);
    __device__ void swap_index(int i, int j);
private:
    int l;
    long int size;
    struct head_t
    {
        head_t *prev, *next;    // a circular list
        float *data;
        int len;        // data[0,len) is cached in this entry
    };

    head_t *head;
    head_t lru_head;
    __device__ void lru_delete(head_t *h);
    __device__ void lru_insert(head_t *h);
};

__device__ CUDA_Cache::CUDA_Cache(int l_,long int size_):l(l_),size(size_)
{
    head = (head_t *)malloc(sizeof(head_t) * l);    // initialized to 0
    memset(head, 0, sizeof(head_t) *l);
    size /= sizeof(float);
    size -= l * sizeof(head_t) / sizeof(float);
    size = max(size, 2 * (long int) l); // cache must be large enough for two columns
    lru_head.next = lru_head.prev = &lru_head;
}

__device__ CUDA_Cache::~CUDA_Cache()
{
    for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
        free(h->data);
    free(head);
}

__device__ void CUDA_Cache::lru_delete(head_t *h)
{
    // delete from current location
    h->prev->next = h->next;
    h->next->prev = h->prev;
}

__device__ void CUDA_Cache::lru_insert(head_t *h)
{
    // insert to last position
    h->next = &lru_head;
    h->prev = lru_head.prev;
    h->prev->next = h;
    h->next->prev = h;
}

__device__ int CUDA_Cache::get_data(const int index, float **data, int len)
{
    head_t *h = &head[index];
    if(h->len) lru_delete(h);
    int more = len - h->len;
    
    if (more > 0)
    {
        // free old space
        while(size < more)
        {
            head_t *old = lru_head.next;
            lru_delete(old);
            free(old->data);
            size += old->len;
            old->data = 0;
            old->len = 0;
        }
        
        // allocate new space
        float *tp = h->data;
        h->data = (float *)malloc(sizeof(float)*len);
        if (tp != NULL)
        {
            memcpy(h->data, tp, sizeof(float)*h->len);
            free(tp);
        }
        size -= more;
        swap(h->len,len);
    }

    lru_insert(h);
    *data = h->data;
    return len;
}

__device__ void CUDA_Cache::swap_index(int i, int j)
{
    if(i==j) return;

    if(head[i].len) lru_delete(&head[i]);
    if(head[j].len) lru_delete(&head[j]);
    swap(head[i].data,head[j].data);
    swap(head[i].len,head[j].len);
    if(head[i].len) lru_insert(&head[i]);
    if(head[j].len) lru_insert(&head[j]);

    if(i>j) swap(i,j);
    for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
    {
        if(h->len > i)
        {
            if(h->len > j)
                swap(h->data[i],h->data[j]);
            else
            {
                // give up
                lru_delete(h);
                free(h->data);
                size += h->len;
                h->data = 0;
                h->len = 0;
            }
        }
    }
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class CUDA_QMatrix {
public:
    __device__ virtual float *get_Q(int column, int len) const = 0;
    __device__ virtual float *get_QD() const = 0;
    __device__ virtual void swap_index(int i, int j) const = 0;
    __device__ virtual ~CUDA_QMatrix() {}
};

class CUDA_Kernel: public CUDA_QMatrix {
public:
    __device__ CUDA_Kernel(int l, svm_node * const * x, const svm_parameter& param);
    __device__ virtual ~CUDA_Kernel();

    __device__ static float k_function(const svm_node *x, const svm_node *y,
                                        const svm_parameter& param);
    __device__ virtual float *get_Q(int column, int len) const = 0;
    __device__ virtual float *get_QD() const = 0;
    __device__ virtual void swap_index(int i, int j) const  // no so const...
    {
        swap(x[i],x[j]);
        if(x_square) swap(x_square[i],x_square[j]);
    }
protected:

    float (CUDA_Kernel::*kernel_function)(int i, int j) const;

private:
    const svm_node **x;
    float *x_square;

    // svm_parameter
    const int kernel_type;
    const int degree;
    const float gamma;
    const float coef0;

    __device__ static float dot(const svm_node *px, const svm_node *py);
    __device__ float kernel_linear(int i, int j) const
    {
        return dot(x[i],x[j]);
    }
    __device__ float kernel_poly(int i, int j) const
    {
        return powi(gamma*dot(x[i],x[j])+coef0,degree);
    }
    __device__ float kernel_rbf(int i, int j) const
    {
        return expf(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
    }
    __device__ float kernel_sigmoid(int i, int j) const
    {
        return tanh(gamma*dot(x[i],x[j])+coef0);
    }
    __device__ float kernel_precomputed(int i, int j) const
    {
        return x[i][(int)(x[j][0].value)].value;
    }
};

__device__ CUDA_Kernel::CUDA_Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
    switch(kernel_type)
    {
        case LINEAR:
            kernel_function = &CUDA_Kernel::kernel_linear;
            break;
        case POLY:
            kernel_function = &CUDA_Kernel::kernel_poly;
            break;
        case RBF:
            kernel_function = &CUDA_Kernel::kernel_rbf;
            break;
        case SIGMOID:
            kernel_function = &CUDA_Kernel::kernel_sigmoid;
            break;
        case PRECOMPUTED:
            kernel_function = &CUDA_Kernel::kernel_precomputed;
            break;
    }

    clone(x,x_,l);

    if(kernel_type == RBF)
    {
        x_square = new float[l];
        for(int i=0;i<l;i++)
            x_square[i] = dot(x[i],x[i]);
    }
    else
        x_square = 0;
}

__device__ CUDA_Kernel::~CUDA_Kernel()
{
    delete[] x;
    delete[] x_square;
}

__device__ float CUDA_Kernel::dot(const svm_node *px, const svm_node *py)
{
    float sum = 0;
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
        }           
    }
    return sum;
}

__device__ float CUDA_Kernel::k_function(const svm_node *x, const svm_node *y,
              const svm_parameter& param)
{
    switch(param.kernel_type)
    {
        case LINEAR:
            return dot(x,y);
        case POLY:
            return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
        case RBF:
        {
            float sum = 0;
            while(x->index != -1 && y->index !=-1)
            {
                if(x->index == y->index)
                {
                    float d = x->value - y->value;
                    sum += d*d;
                    ++x;
                    ++y;
                }
                else
                {
                    if(x->index > y->index)
                    {   
                        sum += y->value * y->value;
                        ++y;
                    }
                    else
                    {
                        sum += x->value * x->value;
                        ++x;
                    }
                }
            }

            while(x->index != -1)
            {
                sum += x->value * x->value;
                ++x;
            }

            while(y->index != -1)
            {
                sum += y->value * y->value;
                ++y;
            }
            
            return expf(-param.gamma*sum);
        }
        case SIGMOID:
            return tanhf(param.gamma*dot(x,y)+param.coef0);
        case PRECOMPUTED:  //x: test (validation), y: SV
            return x[(int)(y->value)].value;
        default:
            return 0;  // Unreachable 
    }
}




// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//  min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//      y^T \alpha = \delta
//      y_i = +1 or -1
//      0 <= alpha_i <= Cp for y_i = 1
//      0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, p, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class CUDA_Solver {
public:
    __device__ CUDA_Solver() {};
    __device__ virtual ~CUDA_Solver() {};

    struct SolutionInfo {
        float obj;
        float rho;
        float upper_bound_p;
        float upper_bound_n;
        float r;   // for CUDA_Solver_NU
    };

    __device__ void Solve(int l, const CUDA_QMatrix& Q, const float *p_, const schar *y_,
                          float *alpha_, float Cp, float Cn, float eps,
                          SolutionInfo* si, int shrinking);
protected:
    int active_size;
    schar *y;
    float *G;      // gradient of objective function
    enum { LOWER_BOUND, UPPER_BOUND, FREE };
    char *alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
    float *alpha;
    const CUDA_QMatrix *Q;
    const float *QD;
    float eps;
    float Cp,Cn;
    float *p;
    int *active_set;
    float *G_bar;      // gradient, if we treat free variables as 0
    int l;
    bool unshrink;  // XXX

    __device__ float get_C(int i)
    {
        return (y[i] > 0)? Cp : Cn;
    }
    __device__ void update_alpha_status(int i)
    {
        if(alpha[i] >= get_C(i))
            alpha_status[i] = UPPER_BOUND;
        else if(alpha[i] <= 0)
            alpha_status[i] = LOWER_BOUND;
        else alpha_status[i] = FREE;
    }
    __device__ bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
    __device__ bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
    __device__ bool is_free(int i) { return alpha_status[i] == FREE; }
    __device__ void swap_index(int i, int j);
    __device__ void reconstruct_gradient();
    __device__ virtual int select_working_set(int &i, int &j);
    __device__ virtual float calculate_rho();
    __device__ virtual void do_shrinking();
private:
    __device__ bool be_shrunk(int i, float Gmax1, float Gmax2);
};

__device__ void CUDA_Solver::swap_index(int i, int j)
{
    Q->swap_index(i,j);
    swap(y[i],y[j]);
    swap(G[i],G[j]);
    swap(alpha_status[i],alpha_status[j]);
    swap(alpha[i],alpha[j]);
    swap(p[i],p[j]);
    swap(active_set[i],active_set[j]);
    swap(G_bar[i],G_bar[j]);
}

__device__ void CUDA_Solver::reconstruct_gradient()
{
    // reconstruct inactive elements of G from G_bar and free variables

    if(active_size == l) return;

    int i,j;
    int nr_free = 0;

    for(j=active_size;j<l;j++)
        G[j] = G_bar[j] + p[j];

    for(j=0;j<active_size;j++)
        if(is_free(j))
            nr_free++;

    if(2*nr_free < active_size)
        printf("\nWARNING: using -h 0 may be faster\n");

    if (nr_free*l > 2*active_size*(l-active_size))
    {
        for(i=active_size;i<l;i++)
        {
            const float *Q_i = Q->get_Q(i,active_size);
            for(j=0;j<active_size;j++)
                if(is_free(j))
                    G[i] += alpha[j] * Q_i[j];
        }
    }
    else
    {
        for(i=0;i<active_size;i++)
            if(is_free(i))
            {
                const float *Q_i = Q->get_Q(i,l);
                float alpha_i = alpha[i];
                for(j=active_size;j<l;j++)
                    G[j] += alpha_i * Q_i[j];
            }
    }
}

__device__ void CUDA_Solver::Solve(int l, const CUDA_QMatrix& Q, const float *p_, const schar *y_,
           float *alpha_, float Cp, float Cn, float eps,
           SolutionInfo* si, int shrinking)
{
    this->l = l;
    this->Q = &Q;
    QD=Q.get_QD();
    clone(p, p_,l);
    clone(y, y_,l);
    clone(alpha,alpha_,l);
    this->Cp = Cp;
    this->Cn = Cn;
    this->eps = eps;
    unshrink = false;

    // initialize alpha_status
    {
        alpha_status = (char *)malloc(sizeof(char) * l);
        for(int i=0;i<l;i++)
            update_alpha_status(i);
    }
    
    // initialize active set (for shrinking)
    {
        active_set = (int *)malloc(sizeof(int) * l);;
        for(int i=0;i<l;i++)
            active_set[i] = i;
        active_size = l;
    }
    
    // initialize gradient
    {
        G = (float *)malloc(sizeof(float) * l);
        G_bar = (float *)malloc(sizeof(float) * l);
        int i;
        for(i=0;i<l;i++)
        {
            G[i] = p[i];
            G_bar[i] = 0;
        }
        
        
        for(i=0;i<l;i++)
            if(!is_lower_bound(i))
            {
                const float *Q_i = Q.get_Q(i,l);
                float alpha_i = alpha[i];
                int j;
                
                for(j=0;j<l;j++)
                    G[j] += alpha_i*Q_i[j];
                if(is_upper_bound(i))
                    for(j=0;j<l;j++)
                        G_bar[j] += get_C(i) * Q_i[j];
            }
    }
    
    // optimization step
    
    int iter = 0;
    int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
    int counter = min(l,1000)+1;
    
    while(iter < max_iter)
    {
        // show progress and do shrinking

        if(--counter == 0)
        {
            counter = min(l,1000);
            if(shrinking) do_shrinking();
            printf(".");
        }
        
        int i,j;
        if (select_working_set(i,j)!=0)
        {
            // reconstruct the whole gradient
            reconstruct_gradient();
            // reset active set size and check
            active_size = l;
            printf("*");
            if(select_working_set(i,j)!=0)
                break;
            else
                counter = 1;    // do shrinking next iteration
        }
        
        ++iter;
        
        // update alpha[i] and alpha[j], handle bounds carefully
        
        const float *Q_i = Q.get_Q(i,active_size);
        const float *Q_j = Q.get_Q(j,active_size);

        float C_i = get_C(i);
        float C_j = get_C(j);

        float old_alpha_i = alpha[i];
        float old_alpha_j = alpha[j];

        if(y[i]!=y[j])
        {
            float quad_coef = QD[i]+QD[j]+2*Q_i[j];
            if (quad_coef <= 0)
                quad_coef = TAU;
            float delta = (-G[i]-G[j])/quad_coef;
            float diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;
            
            if(diff > 0)
            {
                if(alpha[j] < 0)
                {
                    alpha[j] = 0;
                    alpha[i] = diff;
                }
            }
            else
            {
                if(alpha[i] < 0)
                {
                    alpha[i] = 0;
                    alpha[j] = -diff;
                }
            }
            if(diff > C_i - C_j)
            {
                if(alpha[i] > C_i)
                {
                    alpha[i] = C_i;
                    alpha[j] = C_i - diff;
                }
            }
            else
            {
                if(alpha[j] > C_j)
                {
                    alpha[j] = C_j;
                    alpha[i] = C_j + diff;
                }
            }
        }
        else
        {
            float quad_coef = QD[i]+QD[j]-2*Q_i[j];
            if (quad_coef <= 0)
                quad_coef = TAU;
            float delta = (G[i]-G[j])/quad_coef;
            float sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;

            if(sum > C_i)
            {
                if(alpha[i] > C_i)
                {
                    alpha[i] = C_i;
                    alpha[j] = sum - C_i;
                }
            }
            else
            {
                if(alpha[j] < 0)
                {
                    alpha[j] = 0;
                    alpha[i] = sum;
                }
            }
            if(sum > C_j)
            {
                if(alpha[j] > C_j)
                {
                    alpha[j] = C_j;
                    alpha[i] = sum - C_j;
                }
            }
            else
            {
                if(alpha[i] < 0)
                {
                    alpha[i] = 0;
                    alpha[j] = sum;
                }
            }
        }

        // update G

        float delta_alpha_i = alpha[i] - old_alpha_i;
        float delta_alpha_j = alpha[j] - old_alpha_j;
        
        for(int k=0;k<active_size;k++)
        {
            G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
        }

        // update alpha_status and G_bar

        {
            bool ui = is_upper_bound(i);
            bool uj = is_upper_bound(j);
            update_alpha_status(i);
            update_alpha_status(j);
            int k;
            if(ui != is_upper_bound(i))
            {
                Q_i = Q.get_Q(i,l);
                if(ui)
                    for(k=0;k<l;k++)
                        G_bar[k] -= C_i * Q_i[k];
                else
                    for(k=0;k<l;k++)
                        G_bar[k] += C_i * Q_i[k];
            }

            if(uj != is_upper_bound(j))
            {
                Q_j = Q.get_Q(j,l);
                if(uj)
                    for(k=0;k<l;k++)
                        G_bar[k] -= C_j * Q_j[k];
                else
                    for(k=0;k<l;k++)
                        G_bar[k] += C_j * Q_j[k];
            }
        }
    }

    if(iter >= max_iter)
    {
        if(active_size < l)
        {
            // reconstruct the whole gradient to calculate objective value
            reconstruct_gradient();
            active_size = l;
            printf("*");
        }
        printf("\nWARNING: reaching max number of iterations\n");
    }

    // calculate rho

    si->rho = calculate_rho();

    // calculate objective value
    {
        float v = 0;
        int i;
        for(i=0;i<l;i++)
            v += alpha[i] * (G[i] + p[i]);

        si->obj = v/2;
    }

    // put back the solution
    {
        for(int i=0;i<l;i++)
            alpha_[active_set[i]] = alpha[i];
    }

    // juggle everything back
    /*{
        for(int i=0;i<l;i++)
            while(active_set[i] != i)
                swap_index(i,active_set[i]);
                // or Q.swap_index(i,active_set[i]);
    }*/

    si->upper_bound_p = Cp;
    si->upper_bound_n = Cn;

    printf("\noptimization finished, #iter = %d\n",iter);

    free(p);
    free(y);
    free(alpha);
    free(alpha_status);
    free(active_set);
    free(G);
    free(G_bar);
}

// return 1 if already optimal, return 0 otherwise
__device__ int CUDA_Solver::select_working_set(int &out_i, int &out_j)
{
    // return i,j such that
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
    
    float Gmax = -INF;
    float Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    float obj_diff_min = INF;

    for(int t=0;t<active_size;t++)
        if(y[t]==+1)    
        {
            if(!is_upper_bound(t))
                if(-G[t] >= Gmax)
                {
                    Gmax = -G[t];
                    Gmax_idx = t;
                }
        }
        else
        {
            if(!is_lower_bound(t))
                if(G[t] >= Gmax)
                {
                    Gmax = G[t];
                    Gmax_idx = t;
                }
        }
    
    int i = Gmax_idx;
    const float *Q_i = NULL;
    if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
        Q_i = Q->get_Q(i,active_size);
    
    
    for(int j=0;j<active_size;j++)
    {
        if(y[j]==+1)
        {
            if (!is_lower_bound(j))
            {
                float grad_diff=Gmax+G[j];
                if (G[j] >= Gmax2)
                    Gmax2 = G[j];
                if (grad_diff > 0)
                {
                    float obj_diff;
                    float quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else
        {
            if (!is_upper_bound(j))
            {
                float grad_diff= Gmax-G[j];
                if (-G[j] >= Gmax2)
                    Gmax2 = -G[j];
                if (grad_diff > 0)
                {
                    float obj_diff;
                    float quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    if(Gmax+Gmax2 < eps)
        return 1;

    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return 0;
}

__device__ bool CUDA_Solver::be_shrunk(int i, float Gmax1, float Gmax2)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return(-G[i] > Gmax1);
        else
            return(-G[i] > Gmax2);
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return(G[i] > Gmax2);
        else    
            return(G[i] > Gmax1);
    }
    else
        return(false);
}

__device__ void CUDA_Solver::do_shrinking()
{
    int i;
    float Gmax1 = -INF;        // max { -y_i * grad(f)_i | i in I_up(\alpha) }
    float Gmax2 = -INF;        // max { y_i * grad(f)_i | i in I_low(\alpha) }

    // find maximal violating pair first
    for(i=0;i<active_size;i++)
    {
        if(y[i]==+1)    
        {
            if(!is_upper_bound(i))  
            {
                if(-G[i] >= Gmax1)
                    Gmax1 = -G[i];
            }
            if(!is_lower_bound(i))  
            {
                if(G[i] >= Gmax2)
                    Gmax2 = G[i];
            }
        }
        else    
        {
            if(!is_upper_bound(i))  
            {
                if(-G[i] >= Gmax2)
                    Gmax2 = -G[i];
            }
            if(!is_lower_bound(i))  
            {
                if(G[i] >= Gmax1)
                    Gmax1 = G[i];
            }
        }
    }

    if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
    {
        unshrink = true;
        reconstruct_gradient();
        active_size = l;
        printf("*");
    }

    for(i=0;i<active_size;i++)
        if (be_shrunk(i, Gmax1, Gmax2))
        {
            active_size--;
            while (active_size > i)
            {
                if (!be_shrunk(active_size, Gmax1, Gmax2))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size--;
            }
        }
}

__device__ float CUDA_Solver::calculate_rho()
{
    float r;
    int nr_free = 0;
    float ub = INF, lb = -INF, sum_free = 0;
    for(int i=0;i<active_size;i++)
    {
        float yG = y[i]*G[i];

        if(is_upper_bound(i))
        {
            if(y[i]==-1)
                ub = min(ub,yG);
            else
                lb = max(lb,yG);
        }
        else if(is_lower_bound(i))
        {
            if(y[i]==+1)
                ub = min(ub,yG);
            else
                lb = max(lb,yG);
        }
        else
        {
            ++nr_free;
            sum_free += yG;
        }
    }

    if(nr_free>0)
        r = sum_free/nr_free;
    else
        r = (ub+lb)/2;

    return r;
}

//
// CUDA_Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class CUDA_Solver_NU: public CUDA_Solver
{
public:
    __device__ CUDA_Solver_NU() {}
    __device__ void Solve(int l, const CUDA_QMatrix& Q, const float *p, const schar *y,
                          float *alpha, float Cp, float Cn, float eps,
                          SolutionInfo* si, int shrinking)
    {
        this->si = si;
        CUDA_Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
    }
private:
    SolutionInfo *si;
    __device__ int select_working_set(int &i, int &j);
    __device__ float calculate_rho();
    __device__ bool be_shrunk(int i, float Gmax1, float Gmax2, float Gmax3, float Gmax4);
    __device__ void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
__device__ int CUDA_Solver_NU::select_working_set(int &out_i, int &out_j)
{
    // return i,j such that y_i = y_j and
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

    float Gmaxp = -INF;
    float Gmaxp2 = -INF;
    int Gmaxp_idx = -1;

    float Gmaxn = -INF;
    float Gmaxn2 = -INF;
    int Gmaxn_idx = -1;

    int Gmin_idx = -1;
    float obj_diff_min = INF;

    for(int t=0;t<active_size;t++)
        if(y[t]==+1)
        {
            if(!is_upper_bound(t))
                if(-G[t] >= Gmaxp)
                {
                    Gmaxp = -G[t];
                    Gmaxp_idx = t;
                }
        }
        else
        {
            if(!is_lower_bound(t))
                if(G[t] >= Gmaxn)
                {
                    Gmaxn = G[t];
                    Gmaxn_idx = t;
                }
        }

    int ip = Gmaxp_idx;
    int in = Gmaxn_idx;
    const float *Q_ip = NULL;
    const float *Q_in = NULL;
    if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
        Q_ip = Q->get_Q(ip,active_size);
    if(in != -1)
        Q_in = Q->get_Q(in,active_size);

    for(int j=0;j<active_size;j++)
    {
        if(y[j]==+1)
        {
            if (!is_lower_bound(j)) 
            {
                float grad_diff=Gmaxp+G[j];
                if (G[j] >= Gmaxp2)
                    Gmaxp2 = G[j];
                if (grad_diff > 0)
                {
                    float obj_diff;
                    float quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else
        {
            if (!is_upper_bound(j))
            {
                float grad_diff=Gmaxn-G[j];
                if (-G[j] >= Gmaxn2)
                    Gmaxn2 = -G[j];
                if (grad_diff > 0)
                {
                    float obj_diff;
                    float quad_coef = QD[in]+QD[j]-2*Q_in[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
        return 1;

    if (y[Gmin_idx] == +1)
        out_i = Gmaxp_idx;
    else
        out_i = Gmaxn_idx;
    out_j = Gmin_idx;

    return 0;
}

__device__ bool CUDA_Solver_NU::be_shrunk(int i, float Gmax1, float Gmax2, float Gmax3, float Gmax4)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return(-G[i] > Gmax1);
        else    
            return(-G[i] > Gmax4);
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return(G[i] > Gmax2);
        else    
            return(G[i] > Gmax3);
    }
    else
        return(false);
}

__device__ void CUDA_Solver_NU::do_shrinking()
{
    float Gmax1 = -INF;    // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
    float Gmax2 = -INF;    // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
    float Gmax3 = -INF;    // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
    float Gmax4 = -INF;    // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

    // find maximal violating pair first
    int i;
    for(i=0;i<active_size;i++)
    {
        if(!is_upper_bound(i))
        {
            if(y[i]==+1)
            {
                if(-G[i] > Gmax1) Gmax1 = -G[i];
            }
            else    if(-G[i] > Gmax4) Gmax4 = -G[i];
        }
        if(!is_lower_bound(i))
        {
            if(y[i]==+1)
            {   
                if(G[i] > Gmax2) Gmax2 = G[i];
            }
            else    if(G[i] > Gmax3) Gmax3 = G[i];
        }
    }

    if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
    {
        unshrink = true;
        reconstruct_gradient();
        active_size = l;
    }

    for(i=0;i<active_size;i++)
        if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
        {
            active_size--;
            while (active_size > i)
            {
                if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size--;
            }
        }
}

__device__ float CUDA_Solver_NU::calculate_rho()
{
    int nr_free1 = 0,nr_free2 = 0;
    float ub1 = INF, ub2 = INF;
    float lb1 = -INF, lb2 = -INF;
    float sum_free1 = 0, sum_free2 = 0;

    for(int i=0;i<active_size;i++)
    {
        if(y[i]==+1)
        {
            if(is_upper_bound(i))
                lb1 = max(lb1,G[i]);
            else if(is_lower_bound(i))
                ub1 = min(ub1,G[i]);
            else
            {
                ++nr_free1;
                sum_free1 += G[i];
            }
        }
        else
        {
            if(is_upper_bound(i))
                lb2 = max(lb2,G[i]);
            else if(is_lower_bound(i))
                ub2 = min(ub2,G[i]);
            else
            {
                ++nr_free2;
                sum_free2 += G[i];
            }
        }
    }

    float r1,r2;
    if(nr_free1 > 0)
        r1 = sum_free1/nr_free1;
    else
        r1 = (ub1+lb1)/2;
    
    if(nr_free2 > 0)
        r2 = sum_free2/nr_free2;
    else
        r2 = (ub2+lb2)/2;
    
    si->r = (r1+r2)/2;
    return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class CUDA_SVC_Q: public CUDA_Kernel
{ 
public:
    __device__  CUDA_SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
    :CUDA_Kernel(prob.l, prob.x, param)
    {
        clone(y,y_,prob.l);
        cache = new CUDA_Cache(prob.l,(long int)(param.cache_size*(1<<20)));
        QD = (float *)malloc(sizeof(float) * prob.l);
        for(int i=0;i<prob.l;i++)
            QD[i] = (this->*kernel_function)(i,i);
    }
    
    __device__ float *get_Q(int i, int len) const
    {
        float *data;
        int start, j;
        
        if((start = cache->get_data(i,&data,len)) < len)
        {
            for(j=start;j<len;j++)
                data[j] = (float)(y[i]*y[j]*(this->*kernel_function)(i,j));
        }
        return data;
    }

    __device__ float *get_QD() const
    {
        return QD;
    }

    __device__ void swap_index(int i, int j) const
    {
        cache->swap_index(i,j);
        CUDA_Kernel::swap_index(i,j);
        swap(y[i],y[j]);
        swap(QD[i],QD[j]);
    }

    __device__ ~CUDA_SVC_Q()
    {
        free(y);
        delete cache;
        free(QD);
    }
private:
    schar *y;
    CUDA_Cache *cache;
    float *QD;
};

class CUDA_ONE_CLASS_Q: public CUDA_Kernel
{
public:
    __device__ CUDA_ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
    :CUDA_Kernel(prob.l, prob.x, param)
    {
        cache = new CUDA_Cache(prob.l,(long int)(param.cache_size*(1<<20)));
        QD = (float *)malloc(sizeof(float) * prob.l);
        for(int i=0;i<prob.l;i++)
            QD[i] = (this->*kernel_function)(i,i);
    }
    
    __device__ float *get_Q(int i, int len) const
    {
        float *data;
        int start, j;
        if((start = cache->get_data(i,&data,len)) < len)
        {
            for(j=start;j<len;j++)
                data[j] = (float)(this->*kernel_function)(i,j);
        }
        return data;
    }

    __device__ float *get_QD() const
    {
        return QD;
    }

    __device__ void swap_index(int i, int j) const
    {
        cache->swap_index(i,j);
        CUDA_Kernel::swap_index(i,j);
        swap(QD[i],QD[j]);
    }

    __device__ ~CUDA_ONE_CLASS_Q()
    {
        delete cache;
        free(QD);
    }
private:
    CUDA_Cache *cache;
    float *QD;
};

class CUDA_SVR_Q: public CUDA_Kernel
{ 
public:
    __device__ CUDA_SVR_Q(const svm_problem& prob, const svm_parameter& param)
    :CUDA_Kernel(prob.l, prob.x, param)
    {
        l = prob.l;
        cache = new CUDA_Cache(l,(long int)(param.cache_size*(1<<20)));
        QD = new float[2*l];
        sign = new schar[2*l];
        index = new int[2*l];
        for(int k=0;k<l;k++)
        {
            sign[k] = 1;
            sign[k+l] = -1;
            index[k] = k;
            index[k+l] = k;
            QD[k] = (this->*kernel_function)(k,k);
            QD[k+l] = QD[k];
        }
        buffer[0] = new float[2*l];
        buffer[1] = new float[2*l];
        next_buffer = 0;
    }

    __device__ void swap_index(int i, int j) const
    {
        swap(sign[i],sign[j]);
        swap(index[i],index[j]);
        swap(QD[i],QD[j]);
    }
    
    __device__ float *get_Q(int i, int len) const
    {
        float *data;
        int j, real_i = index[i];
        if(cache->get_data(real_i,&data,l) < l)
        {
            for(j=0;j<l;j++)
                data[j] = (float)(this->*kernel_function)(real_i,j);
        }

        // reorder and copy
        float *buf = buffer[next_buffer];
        next_buffer = 1 - next_buffer;
        schar si = sign[i];
        for(j=0;j<len;j++)
            buf[j] = (float) si * (float) sign[j] * data[index[j]];
        return buf;
    }

    __device__ float *get_QD() const
    {
        return QD;
    }

    __device__ ~CUDA_SVR_Q()
    {
        delete cache;
        delete[] sign;
        delete[] index;
        delete[] buffer[0];
        delete[] buffer[1];
        delete[] QD;
    }
private:
    int l;
    CUDA_Cache *cache;
    schar *sign;
    int *index;
    mutable int next_buffer;
    float *buffer[2];
    float *QD;
};





__device__ void cuda_svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = (int *)malloc(sizeof(int) * max_nr_class);
    int *count = (int *)malloc(sizeof(int) * max_nr_class);
    int *data_label = (int *)malloc(sizeof(int) * l);
    int i;

    for(i=0;i<l;i++)
    {
        int this_label = (int)prob->y[i];
        int j;
        for(j=0;j<nr_class;j++)
        {
            if(this_label == label[j])
            {
                ++count[j];
                break;
            }
        }
        data_label[i] = j;
        if(j == nr_class)
        {
            if(nr_class == max_nr_class)
            {
                int *label_t = label;
                int *count_t = count;
                label = (int *)malloc(2 * max_nr_class * sizeof(int));
                count = (int *)malloc(2 * max_nr_class * sizeof(int));
                int k;
                for (k=0; k<max_nr_class; k++)
                {
                    label[k] = label_t[k];
                    count[k] = count_t[k];
                }
                free(label_t);
                free(count_t);
                max_nr_class *= 2;
            }
            label[nr_class] = this_label;
            count[nr_class] = 1;
            ++nr_class;
        }
    }

    //
    // Labels are ordered by their first occurrence in the training set. 
    // However, for two-class sets with -1/+1 labels and -1 appears first, 
    // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
    //
    if (nr_class == 2 && label[0] == -1 && label[1] == 1)
    {
        swap(label[0],label[1]);
        swap(count[0],count[1]);
        for(i=0;i<l;i++)
        {
            if(data_label[i] == 0)
                data_label[i] = 1;
            else
                data_label[i] = 0;
        }
    }

    int *start = (int *)malloc(sizeof(int) * nr_class);
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];
    for(i=0;i<l;i++)
    {
        perm[start[data_label[i]]] = i;
        ++start[data_label[i]];
    }
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];

    *nr_class_ret = nr_class;
    *label_ret = label;
    *start_ret = start;
    *count_ret = count;
    free(data_label);
}


__host__ int svm_get_nr_classes(const svm_problem *prob)
{
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = (int *)malloc(sizeof(int) * max_nr_class);
    int i;

    for(i=0;i<l;i++)
    {
        int this_label = (int)prob->y[i];
        int j;
        for(j=0;j<nr_class;j++)
            if(this_label == label[j])
                break;
        if(j == nr_class)
        {
            if(nr_class == max_nr_class)
            {
                max_nr_class *= 2;
                label = (int *)realloc(label,max_nr_class*sizeof(int));
            }
            label[nr_class] = this_label;
            ++nr_class;
        }
    }
    
    return nr_class;
}




__device__ void cuda_svm_free_model_content(svm_model* model_ptr)
{
    if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
        free((void *)(model_ptr->SV[0]));
    if(model_ptr->sv_coef)
    {
        for(int i=0;i<model_ptr->nr_class-1;i++)
            free(model_ptr->sv_coef[i]);
    }

    free(model_ptr->SV);
    model_ptr->SV = NULL;

    free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;

    free(model_ptr->rho);
    model_ptr->rho = NULL;

    free(model_ptr->label);
    model_ptr->label= NULL;

    free(model_ptr->probA);
    model_ptr->probA = NULL;

    free(model_ptr->probB);
    model_ptr->probB= NULL;

    free(model_ptr->sv_indices);
    model_ptr->sv_indices = NULL;

    free(model_ptr->nSV);
    model_ptr->nSV = NULL;
}

__device__ void cuda_svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
    if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
    {
        cuda_svm_free_model_content(*model_ptr_ptr);
        free(*model_ptr_ptr);
        *model_ptr_ptr = NULL;
    }
}

__device__ void cuda_svm_destroy_param(svm_parameter* param)
{
    free(param->weight_label);
    free(param->weight);
}





// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
__device__ void cuda_sigmoid_train(int l, const float *dec_values, const float *labels, float& A, float& B)
{
    float prior1=0, prior0 = 0;
    int i;

    for (i=0;i<l;i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;
    
    int max_iter=100;    // Maximal number of iterations
    float min_step=1e-10;    // Minimal step taken in line search
    float sigma=1e-12;    // For numerically strict PD of Hessian
    float eps=1e-5;
    float hiTarget=(prior1+1.0)/(prior1+2.0);
    float loTarget=1.0/(prior0+2.0);
    float *t=(float *)malloc(sizeof(float) * l);
    float fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    float newA,newB,newf,d1,d2;
    int iter;
    
    // Initial Point and Initial Fun Value
    A=0.0; B=logf((prior0+1.0)/(prior1+1.0));
    float fval = 0.0;

    for (i=0;i<l;i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + logf(1+expf(-fApB));
        else
            fval += (t[i] - 1)*fApB +logf(1+expf(fApB));
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma; // numerically ensures strict PD
        h22=sigma;
        h21=0.0;g1=0.0;g2=0.0;
        for (i=0;i<l;i++)
        {
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                q=1.0/(1.0+expf(-fApB));
                p=expf(-fApB)*q;
            }
            else
            {
                p=1.0/(1.0+expf(fApB));
                q=expf(fApB)*p;
            }
            d2=p*q;
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }

        // Stopping Criteria
        if (fabs(g1)<eps && fabs(g2)<eps)
            break;

        // Finding Newton direction: -inv(H') * g
        det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;


        stepsize = 1;        // Line Search
        while (stepsize >= min_step)
        {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i=0;i<l;i++)
            {
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + logf(1+expf(-fApB));
                else
                    newf += (t[i] - 1)*fApB +logf(1+expf(fApB));
            }
            // Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd)
            {
                A=newA;B=newB;fval=newf;
                break;
            }
            else
                stepsize = stepsize / 2.0;
        }

        if (stepsize < min_step)
        {
            printf("Line search fails in two-class probability estimates\n");
            break;
        }
    }
    
    if (iter>=max_iter)
        printf("Reaching maximal iterations in two-class probability estimates\n");
    free(t);
}




__device__ float cuda_sigmoid_predict(float decision_value, float A, float B)
{
    float fApB = decision_value*A+B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return expf(-fApB)/(1.0+expf(-fApB));
    else
        return 1.0/(1+expf(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
__device__ void cuda_multiclass_probability(int k, float **r, float *p)
{
    int t,j;
    int iter = 0, max_iter=max(100,k);
    float **Q=(float **)malloc(sizeof(float *) * k);
    float *Qp=(float *)malloc(sizeof(float) * k);
    float pQp, eps=0.005/k;
    
    for (t=0;t<k;t++)
    {
        p[t]=1.0/k;  // Valid if k = 1
        Q[t]=(float *)malloc(sizeof(float) * k);
        Q[t][t]=0;
        for (j=0;j<t;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=Q[j][t];
        }
        for (j=t+1;j<k;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=-r[j][t]*r[t][j];
        }
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0;
        for (t=0;t<k;t++)
        {
            Qp[t]=0;
            for (j=0;j<k;j++)
                Qp[t]+=Q[t][j]*p[j];
            pQp+=p[t]*Qp[t];
        }
        float max_error=0;
        for (t=0;t<k;t++)
        {
            float error=fabs(Qp[t]-pQp);
            if (error>max_error)
                max_error=error;
        }
        if (max_error<eps) break;
        
        for (t=0;t<k;t++)
        {
            float diff=(-Qp[t]+pQp)/Q[t][t];
            p[t]+=diff;
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
            for (j=0;j<k;j++)
            {
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
                p[j]/=(1+diff);
            }
        }
    }
    if (iter>=max_iter)
        printf("Exceeds max_iter in multiclass_prob\n");
    for(t=0;t<k;t++) free(Q[t]);
    free(Q);
    free(Qp);
}






// Cross-validation decision values for probability estimates
__device__ void cuda_svm_binary_svc_probability(const svm_problem *prob, const svm_parameter *param, float Cp, float Cn, float& probA, float& probB)
{
    int i;
    int nr_fold = 5;
    int *perm = (int *)malloc(sizeof(int) * prob->l);
    float *dec_values = (float *)malloc(sizeof(float) * prob->l);
    CUDA_Rand rand;
    
    // random shuffle
    for(i=0;i<prob->l;i++) perm[i]=i;
    for(i=0;i<prob->l;i++)
    {
        int j = i+rand.rand_int(prob->l-i);
        swap(perm[i],perm[j]);
    }
    for(i=0;i<nr_fold;i++)
    {
        int begin = i*prob->l/nr_fold;
        int end = (i+1)*prob->l/nr_fold;
        int j,k;
        struct svm_problem subprob;

        subprob.l = prob->l-(end-begin);
        subprob.x = (struct svm_node **)malloc(sizeof(struct svm_node*) * subprob.l);
        subprob.y = (float *)malloc(sizeof(float) * subprob.l);
            
        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end;j<prob->l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        int p_count=0,n_count=0;
        for(j=0;j<k;j++)
            if(subprob.y[j]>0)
                p_count++;
            else
                n_count++;

        if(p_count==0 && n_count==0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 0;
        else if(p_count > 0 && n_count == 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 1;
        else if(p_count == 0 && n_count > 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = -1;
        else
        {
            svm_parameter subparam = *param;
            subparam.probability=0;
            subparam.C=1.0;
            subparam.nr_weight=2;
            subparam.weight_label = (int *)malloc(sizeof(int) * 2);
            subparam.weight = (float *)malloc(sizeof(float) * 2);
            subparam.weight_label[0]=+1;
            subparam.weight_label[1]=-1;
            subparam.weight[0]=Cp;
            subparam.weight[1]=Cn;
            struct svm_model *submodel = cuda_device_svm_train_no_prob(&subprob, &subparam);
            for(j=begin;j<end;j++)
            {
                cuda_svm_predict_values(submodel, prob->x[perm[j]], &(dec_values[perm[j]]));
                // ensure +1 -1 order; reason not using CV subroutine
                dec_values[perm[j]] *= submodel->label[0];
            }        
            cuda_svm_free_and_destroy_model(&submodel);
            cuda_svm_destroy_param(&subparam);
        }
        free(subprob.x);
        free(subprob.y);
    }
    cuda_sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
    free(dec_values);
    free(perm);
}





__device__ float cuda_svm_svr_probability(const svm_problem *prob, const svm_parameter *param)
{
    int i;
    int nr_fold = 5;
    float *ymv = (float *)malloc(sizeof(float) * prob->l);
    float mae = 0;

    svm_parameter newparam = *param;
    newparam.probability = 0;
    cuda_svm_cross_validation(prob,&newparam,nr_fold,ymv);
    for(i=0;i<prob->l;i++)
    {
        ymv[i]=prob->y[i]-ymv[i];
        mae += fabs(ymv[i]);
    }       
    mae /= prob->l;
    float std=sqrtf(2*mae*mae);
    int count=0;
    mae=0;
    for(i=0;i<prob->l;i++)
        if (fabs(ymv[i]) > 5*std) 
            count=count+1;
        else 
            mae+=fabs(ymv[i]);
    mae /= (prob->l-count);
    printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
    free(ymv);
    return mae;
}




__device__ float cuda_svm_predict_values(const svm_model *model, const svm_node *x, float* dec_values)
{
    int i;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
    {
        float *sv_coef = model->sv_coef[0];
        float sum = 0;
        for(i=0;i<model->l;i++)
            sum += sv_coef[i] * CUDA_Kernel::k_function(x,model->SV[i],model->param);
        sum -= model->rho[0];
        *dec_values = sum;

        if(model->param.svm_type == ONE_CLASS)
            return (sum>0)?1:-1;
        else
            return sum;
    }
    else
    {
        int nr_class = model->nr_class;
        int l = model->l;
        
        float *kvalue = (float *)malloc(sizeof(float) * l);
        for(i=0;i<l;i++)
            kvalue[i] = CUDA_Kernel::k_function(x,model->SV[i],model->param);

        int *start = (int *)malloc(sizeof(int) * nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+model->nSV[i-1];

        int *vote = (int *)malloc(sizeof(int) * nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;

        int p=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                float sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];
                
                int k;
                float *coef1 = model->sv_coef[j-1];
                float *coef2 = model->sv_coef[i];
                for(k=0;k<ci;k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                dec_values[p] = sum;

                if(dec_values[p] > 0)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        int vote_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;

        free(kvalue);
        free(start);
        free(vote);
        return model->label[vote_max_idx];
    }
}




__device__ float cuda_svm_predict(const svm_model *model, const svm_node *x)
{
    int nr_class = model->nr_class;
    float *dec_values;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
        dec_values = (float *)malloc(sizeof(float));
    else 
        dec_values = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
    float pred_result = cuda_svm_predict_values(model, x, dec_values);
    free(dec_values);
    return pred_result;
}



__device__ float cuda_svm_predict_probability(const svm_model *model, const svm_node *x, float *prob_estimates)
{
    if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
        model->probA!=NULL && model->probB!=NULL)
    {
        int i;
        int nr_class = model->nr_class;
        float *dec_values = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
        cuda_svm_predict_values(model, x, dec_values);

        float min_prob=1e-7;
        float **pairwise_prob=(float **)malloc(sizeof(float *) * nr_class);
        for(i=0;i<nr_class;i++)
            pairwise_prob[i]=(float *)malloc(sizeof(float) * nr_class);
        int k=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                pairwise_prob[i][j]=min(max(cuda_sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
                pairwise_prob[j][i]=1-pairwise_prob[i][j];
                k++;
            }
        cuda_multiclass_probability(nr_class,pairwise_prob,prob_estimates);

        int prob_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(prob_estimates[i] > prob_estimates[prob_max_idx])
                prob_max_idx = i;
        for(i=0;i<nr_class;i++)
            free(pairwise_prob[i]);
        free(dec_values);
        free(pairwise_prob);
        return model->label[prob_max_idx];
    }
    else 
        return cuda_svm_predict(model, x);
}



// Stratified cross validation
__device__ void cuda_svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, float *target)
{
    int i;
    int *fold_start;
    int l = prob->l;
    int *perm = (int *)malloc(sizeof(int) * l);
    int nr_class;
    CUDA_Rand rand;
    
    if (nr_fold > l)
    {
        nr_fold = l;
        printf("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
    }
    fold_start = (int *)malloc(sizeof(int) * (nr_fold+1));
    // stratified cv may not give leave-one-out rate
    // Each class to l folds -> some folds may have zero elements
    if((param->svm_type == C_SVC ||
        param->svm_type == NU_SVC) && nr_fold < l)
    {
        int *start = NULL;
        int *label = NULL;
        int *count = NULL;
        cuda_svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

        // random shuffle and then data grouped by fold using the array perm
        int *fold_count = (int *)malloc(sizeof(int) * nr_fold);
        int c;
        int *index = (int *)malloc(sizeof(int) * l);
        for(i=0;i<l;i++)
            index[i]=perm[i];
        for (c=0; c<nr_class; c++) 
            for(i=0;i<count[c];i++)
            {
                int j = i+rand.rand_int(count[c]-i);
                swap(index[start[c]+j],index[start[c]+i]);
            }
        for(i=0;i<nr_fold;i++)
        {
            fold_count[i] = 0;
            for (c=0; c<nr_class;c++)
                fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
        }
        fold_start[0]=0;
        for (i=1;i<=nr_fold;i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        for (c=0; c<nr_class;c++)
            for(i=0;i<nr_fold;i++)
            {
                int begin = start[c]+i*count[c]/nr_fold;
                int end = start[c]+(i+1)*count[c]/nr_fold;
                for(int j=begin;j<end;j++)
                {
                    perm[fold_start[i]] = index[j];
                    fold_start[i]++;
                }
            }
        fold_start[0]=0;
        for (i=1;i<=nr_fold;i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        free(start);
        free(label);
        free(count);
        free(index);
        free(fold_count);
    }
    else
    {
        for(i=0;i<l;i++) perm[i]=i;
        for(i=0;i<l;i++)
        {
            int j = i+rand.rand_int(l-i);
            swap(perm[i],perm[j]);
        }
        for(i=0;i<=nr_fold;i++)
            fold_start[i]=i*l/nr_fold;
    }
    
    
    for(i=0;i<nr_fold;i++)
    {
        int begin = fold_start[i];
        int end = fold_start[i+1];
        int j,k;
        struct svm_problem subprob;

        subprob.l = l-(end-begin);
        subprob.x = (svm_node **)malloc(sizeof(svm_node *) * subprob.l);
        subprob.y = (float *)malloc(sizeof(float) * subprob.l);
            
        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end;j<l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        struct svm_model *submodel = cuda_device_svm_train_no_prob(&subprob, param);
        //if(param->probability && 
        //   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
        //{
        //    float *prob_estimates = (float *)malloc(sizeof(float) * submodel->nr_class);
        //    for(j=begin;j<end;j++)
        //        target[perm[j]] = cuda_svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
        //    free(prob_estimates);
        //}
        //else
            for(j=begin;j<end;j++)
                target[perm[j]] = cuda_svm_predict(submodel,prob->x[perm[j]]);
        cuda_svm_free_and_destroy_model(&submodel);
        free(subprob.x);
        free(subprob.y);
    }        
    free(fold_start);
    free(perm);
}





//
// construct and solve various formulations
//
__device__ void solve_c_svc(const svm_problem *prob, const svm_parameter *param, float *alpha, CUDA_Solver::SolutionInfo* si, float Cp, float Cn)
{
    int l = prob->l;
    float *minus_ones = (float *)malloc(sizeof(float) * l);
    schar *y = (schar *)malloc(sizeof(schar) * l);

    int i;

    for(i=0;i<l;i++)
    {
        alpha[i] = 0;
        minus_ones[i] = -1;
        if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
    }
    
    CUDA_Solver s;
    s.Solve(l, CUDA_SVC_Q(*prob,*param,y), minus_ones, y,
        alpha, Cp, Cn, param->eps, si, param->shrinking);

    float sum_alpha=0;
    for(i=0;i<l;i++)
        sum_alpha += alpha[i];

    //if (Cp==Cn)
    //  printf("nu = %f\n", sum_alpha/(Cp*prob->l));

    for(i=0;i<l;i++)
        alpha[i] *= y[i];

    free(minus_ones);
    free(y);
}

__device__ void solve_nu_svc(const svm_problem *prob, const svm_parameter *param, float *alpha, CUDA_Solver::SolutionInfo* si)
{
    int i;
    int l = prob->l;
    float nu = param->nu;

    schar *y = (schar *)malloc(sizeof(schar) * l);

    for(i=0;i<l;i++)
        if(prob->y[i]>0)
            y[i] = +1;
        else
            y[i] = -1;

    float sum_pos = nu*l/2;
    float sum_neg = nu*l/2;

    for(i=0;i<l;i++)
        if(y[i] == +1)
        {
            alpha[i] = min(1.0,sum_pos);
            sum_pos -= alpha[i];
        }
        else
        {
            alpha[i] = min(1.0,sum_neg);
            sum_neg -= alpha[i];
        }

    float *zeros = (float *)malloc(sizeof(float) * l);

    for(i=0;i<l;i++)
        zeros[i] = 0;

    CUDA_Solver_NU s;
    s.Solve(l, CUDA_SVC_Q(*prob,*param,y), zeros, y,
        alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
    float r = si->r;

    //printf("C = %f\n",1/r);

    for(i=0;i<l;i++)
        alpha[i] *= y[i]/r;

    si->rho /= r;
    si->obj /= (r*r);
    si->upper_bound_p = 1/r;
    si->upper_bound_n = 1/r;

    free(y);
    free(zeros);
}

__device__ void solve_one_class(const svm_problem *prob, const svm_parameter *param, float *alpha, CUDA_Solver::SolutionInfo* si)
{
    int l = prob->l;
    float *zeros = (float *)malloc(sizeof(float) * l);
    schar *ones = (schar *)malloc(sizeof(schar) * l);
    int i;

    int n = (int)(param->nu*prob->l);   // # of alpha's at upper bound

    for(i=0;i<n;i++)
        alpha[i] = 1;
    if(n<prob->l)
        alpha[n] = param->nu * prob->l - n;
    for(i=n+1;i<l;i++)
        alpha[i] = 0;

    for(i=0;i<l;i++)
    {
        zeros[i] = 0;
        ones[i] = 1;
    }

    CUDA_Solver s;
    s.Solve(l, CUDA_ONE_CLASS_Q(*prob,*param), zeros, ones,
        alpha, 1.0, 1.0, param->eps, si, param->shrinking);

    free(zeros);
    free(ones);
}

__device__ void solve_epsilon_svr(const svm_problem *prob, const svm_parameter *param, float *alpha, CUDA_Solver::SolutionInfo* si)
{
    int l = prob->l;
    float *alpha2 = (float *)malloc(sizeof(float) * l * 2);
    float *linear_term = (float *)malloc(sizeof(float) * l * 2);
    schar *y = (schar *)malloc(sizeof(schar) * l * 2);
    int i;

    for(i=0;i<l;i++)
    {
        alpha2[i] = 0;
        linear_term[i] = param->p - prob->y[i];
        y[i] = 1;

        alpha2[i+l] = 0;
        linear_term[i+l] = param->p + prob->y[i];
        y[i+l] = -1;
    }

    CUDA_Solver s;
    s.Solve(2*l, CUDA_SVR_Q(*prob,*param), linear_term, y,
        alpha2, param->C, param->C, param->eps, si, param->shrinking);

    float sum_alpha = 0;
    for(i=0;i<l;i++)
    {
        alpha[i] = alpha2[i] - alpha2[i+l];
        sum_alpha += fabs(alpha[i]);
    }
    //printf("nu = %f\n",sum_alpha/(param->C*l));

    free(alpha2);
    free(linear_term);
    free(y);
}

__device__ void solve_nu_svr(const svm_problem *prob, const svm_parameter *param, float *alpha, CUDA_Solver::SolutionInfo* si)
{
    int l = prob->l;
    float C = param->C;
    float *alpha2 = (float *)malloc(sizeof(float) * l * 2);
    float *linear_term = (float *)malloc(sizeof(float) * l * 2);
    schar *y = (schar *)malloc(sizeof(schar) * l * 2);
    int i;

    float sum = C * param->nu * l / 2;
    for(i=0;i<l;i++)
    {
        alpha2[i] = alpha2[i+l] = min(sum,C);
        sum -= alpha2[i];

        linear_term[i] = - prob->y[i];
        y[i] = 1;

        linear_term[i+l] = prob->y[i];
        y[i+l] = -1;
    }

    CUDA_Solver_NU s;
    s.Solve(2*l, CUDA_SVR_Q(*prob,*param), linear_term, y,
        alpha2, C, C, param->eps, si, param->shrinking);

    //printf("epsilon = %f\n",-si->r);

    for(i=0;i<l;i++)
        alpha[i] = alpha2[i] - alpha2[i+l];

    free(alpha2);
    free(linear_term);
    free(y);
}






__device__ struct decision_function cuda_svm_train_one(const svm_problem *prob, const svm_parameter *param, float Cp, float Cn)
{
    float *alpha = (float *)malloc(sizeof(float) * prob->l);
    CUDA_Solver::SolutionInfo si;
    switch(param->svm_type)
    {
        case C_SVC:
            solve_c_svc(prob,param,alpha,&si,Cp,Cn);
            break;
        case NU_SVC:
            solve_nu_svc(prob,param,alpha,&si);
            break;
        case ONE_CLASS:
            solve_one_class(prob,param,alpha,&si);
            break;
        case EPSILON_SVR:
            solve_epsilon_svr(prob,param,alpha,&si);
            break;
        case NU_SVR:
            solve_nu_svr(prob,param,alpha,&si);
            break;
    }

    printf("obj = %f, rho = %f\n",si.obj,si.rho);
    
    // output SVs
    /*
    int nSV = 0;
    int nBSV = 0;
    for(int i=0;i<prob->l;i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(prob->y[i] > 0)
            {
                if(fabs(alpha[i]) >= si.upper_bound_p)
                    ++nBSV;
            }
            else
            {
                if(fabs(alpha[i]) >= si.upper_bound_n)
                    ++nBSV;
            }
        }
    }

    printf("nSV = %d, nBSV = %d\n",nSV,nBSV);
    */
    
    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    return f;
}








__device__ struct svm_model *cuda_device_svm_train_no_prob(const struct svm_problem *prob, const struct svm_parameter *param)
{
    svm_model *model = (svm_model *)malloc(sizeof(svm_model));
    model->param = *param;
    model->free_sv = 0;    // XXX

    if(param->svm_type == ONE_CLASS ||
       param->svm_type == EPSILON_SVR ||
       param->svm_type == NU_SVR)
    {
        // regression or one-class-svm
        model->nr_class = 2;
        model->label = NULL;
        model->nSV = NULL;
        model->probA = NULL; model->probB = NULL;
        model->sv_coef = (float **)malloc(sizeof(float*));

        //if(param->probability && 
        //   (param->svm_type == EPSILON_SVR ||
        //    param->svm_type == NU_SVR))
        //{
        //    model->probA = (float *)malloc(sizeof(float));
        //    model->probA[0] = cuda_svm_svr_probability(prob,param);
        //}

        decision_function f = cuda_svm_train_one(prob,param,0,0);
        model->rho = (float *)malloc(sizeof(float));
        model->rho[0] = f.rho;

        int nSV = 0;
        int i;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0) ++nSV;
        model->l = nSV;
        model->SV = (svm_node **)malloc(sizeof(svm_node *) * nSV);
        model->sv_coef[0] = (float *)malloc(sizeof(float) * nSV);
        model->sv_indices = (int *)malloc(sizeof(int) * nSV);
        int j = 0;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0)
            {
                model->SV[j] = prob->x[i];
                model->sv_coef[0][j] = f.alpha[i];
                model->sv_indices[j] = i+1;
                ++j;
            }        

        free(f.alpha);
    }
    else
    {
        // classification
        int l = prob->l;
        int nr_class;
        int *label = NULL;
        int *start = NULL;
        int *count = NULL;
        int *perm = (int *)malloc(sizeof(int) * l);

        // group training data of the same class
        cuda_svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
        if(nr_class == 1) 
            printf("WARNING: training data in only one class. See README for details.\n");
        
        svm_node **x = (svm_node **)malloc(sizeof(svm_node *) * l);
        int i;
        for(i=0;i<l;i++)
            x[i] = prob->x[perm[i]];

        // calculate weighted C

        float *weighted_C = (float *)malloc(sizeof(float) * nr_class);
        for(i=0;i<nr_class;i++)
            weighted_C[i] = param->C;
        for(i=0;i<param->nr_weight;i++)
        {    
            int j;
            for(j=0;j<nr_class;j++)
                if(param->weight_label[i] == label[j])
                    break;
            if(j == nr_class)
                printf("WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }

        // train k*(k-1)/2 models
        
        bool *nonzero = (bool *)malloc(sizeof(bool) * l);
        for(i=0;i<l;i++)
            nonzero[i] = false;
        decision_function *f = (decision_function *)malloc(sizeof(decision_function) * nr_class*(nr_class-1)/2);

        //float *probA=NULL,*probB=NULL;
        //if (param->probability)
        //{
        //    probA=(float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
        //    probB=(float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
        //}

        int p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                svm_problem sub_prob;
                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
                sub_prob.x = (svm_node **)malloc(sizeof(svm_node *) * sub_prob.l);
                sub_prob.y = (float *)malloc(sizeof(float) * sub_prob.l);
                int k;
                for(k=0;k<ci;k++)
                {
                    sub_prob.x[k] = x[si+k];
                    sub_prob.y[k] = +1;
                }
                for(k=0;k<cj;k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];
                    sub_prob.y[ci+k] = -1;
                }

                //if(param->probability)
                //    cuda_svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

                f[p] = cuda_svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
                for(k=0;k<ci;k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                for(k=0;k<cj;k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                free(sub_prob.x);
                free(sub_prob.y);
                ++p;
            }
        
        // build output

        model->nr_class = nr_class;
        
        model->label = (int *)malloc(sizeof(int) * nr_class);
        for(i=0;i<nr_class;i++)
            model->label[i] = label[i];
        
        model->rho = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            model->rho[i] = f[i].rho;

        //if(param->probability)
        //{
        //    model->probA = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
        //    model->probB = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
        //    for(i=0;i<nr_class*(nr_class-1)/2;i++)
        //    {
        //        model->probA[i] = probA[i];
        //        model->probB[i] = probB[i];
        //    }
        //}
        //else
        //{
            model->probA=NULL;
            model->probB=NULL;
        //}

        int total_sv = 0;
        int *nz_count = (int *)malloc(sizeof(int) * nr_class);
        model->nSV = (int *)malloc(sizeof(int) * nr_class);
        for(i=0;i<nr_class;i++)
        {
            int nSV = 0;
            for(int j=0;j<count[i];j++)
                if(nonzero[start[i]+j])
                {    
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }
        
        printf("Total nSV = %d\n",total_sv);

        model->l = total_sv;
        model->SV = (svm_node **)malloc(sizeof(svm_node *) * total_sv);
        model->sv_indices = (int *)malloc(sizeof(int) * total_sv);
        p = 0;
        for(i=0;i<l;i++)
            if(nonzero[i])
            {
                model->SV[p] = x[i];
                model->sv_indices[p++] = perm[i] + 1;
            }

        int *nz_start = (int *)malloc(sizeof(int) * nr_class);
        nz_start[0] = 0;
        for(i=1;i<nr_class;i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];

        model->sv_coef = (float **)malloc(sizeof(float *) * (nr_class-1));
        for(i=0;i<nr_class-1;i++)
            model->sv_coef[i] = (float *)malloc(sizeof(float) * total_sv);

        p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];
                int sj = start[j];
                int ci = count[i];
                int cj = count[j];
                
                int q = nz_start[i];
                int k;
                for(k=0;k<ci;k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];
                q = nz_start[j];
                for(k=0;k<cj;k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];
                ++p;
            }
        
        free(label);
        //free(probA);
        //free(probB);
        free(count);
        free(perm);
        free(start);
        free(x);
        free(weighted_C);
        free(nonzero);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            free(f[i].alpha);
        free(f);
        free(nz_count);
        free(nz_start);
    }
    return model;
}









__device__ void cuda_perform_svm_train(const struct svm_problem *prob, const struct svm_parameter *param, struct svm_model *model)
{
    model->param = *param;
    model->free_sv = 0; // XXX

    if(param->svm_type == ONE_CLASS ||
       param->svm_type == EPSILON_SVR ||
       param->svm_type == NU_SVR)
    {
        // regression or one-class-svm
        model->nr_class = 2;

        if(param->probability && 
           (param->svm_type == EPSILON_SVR ||
            param->svm_type == NU_SVR))
        {
            model->probA[0] = cuda_svm_svr_probability(prob, param);
        }

        decision_function f = cuda_svm_train_one(prob,param,0,0);
        model->rho[0] = f.rho;

        int nSV = 0;
        int i;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0) ++nSV;
        model->l = nSV;
        int j = 0;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0)
            {
                model->SV[j] = (struct svm_node *)i;
                model->sv_coef[0][j] = f.alpha[i];
                model->sv_indices[j] = i+1;
                ++j;
            }       
        
        free(f.alpha);
    }
    else
    {
        // classification
        int l = prob->l;
        int nr_class;
        int *label = NULL;
        int *start = NULL;
        int *count = NULL;
        int *perm = (int *)malloc(sizeof(int) * l);

        // group training data of the same class
        cuda_svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
        if(nr_class == 1) 
            printf("WARNING: training data in only one class. See README for details.\n");
        
        svm_node **x = (svm_node **)malloc(sizeof(svm_node *) * l);
        int i;
        for(i=0;i<l;i++)
            x[i] = prob->x[perm[i]];

        // calculate weighted C
        float *weighted_C = (float *)malloc(sizeof(float) * nr_class);
        for(i=0;i<nr_class;i++)
            weighted_C[i] = param->C;
        for(i=0;i<param->nr_weight;i++)
        {   
            int j;
            for(j=0;j<nr_class;j++)
                if(param->weight_label[i] == label[j])
                    break;
            if(j == nr_class)
                printf("WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }
        
        // train k*(k-1)/2 models
        
        bool *nonzero = (bool *)malloc(sizeof(bool) * l);
        for(i=0;i<l;i++)
            nonzero[i] = false;
        decision_function *f = (decision_function *)malloc(sizeof(decision_function) * nr_class*(nr_class-1)/2);

        float *probA=NULL,*probB=NULL;
        if (param->probability)
        {
            probA=(float *)malloc(sizeof(float) *  nr_class*(nr_class-1)/2);
            probB=(float *)malloc(sizeof(float) *  nr_class*(nr_class-1)/2);
        }
        
        int p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                svm_problem sub_prob;
                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
                sub_prob.x = (svm_node **)malloc(sizeof(svm_node *) * sub_prob.l);
                sub_prob.y = (float *)malloc(sizeof(float) * sub_prob.l);
                int k;
                for(k=0;k<ci;k++)
                {
                    sub_prob.x[k] = x[si+k];
                    sub_prob.y[k] = +1;
                }
                for(k=0;k<cj;k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];
                    sub_prob.y[ci+k] = -1;
                }
                
                if (param->probability)
                    cuda_svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);
                

                f[p] = cuda_svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
                
                for(k=0;k<ci;k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                for(k=0;k<cj;k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                free(sub_prob.x);
                free(sub_prob.y);
                ++p;
            }
        
        // build output
        
        model->nr_class = nr_class;
        
        for(i=0;i<nr_class;i++)
            model->label[i] = label[i];
        
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            model->rho[i] = f[i].rho;

        if(param->probability)
        {
            for(i=0;i<nr_class*(nr_class-1)/2;i++)
            {
                model->probA[i] = probA[i];
                model->probB[i] = probB[i];
            }
        }
        else
        {
            model->probA=NULL;
            model->probB=NULL;
        }

        int total_sv = 0;
        int *nz_count = (int *)malloc(sizeof(int) * nr_class);
        for(i=0;i<nr_class;i++)
        {
            int nSV = 0;
            for(int j=0;j<count[i];j++)
                if(nonzero[start[i]+j])
                {   
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }
        
        printf("Total nSV = %d\n",total_sv);
        
        model->l = total_sv;
        p = 0;
        for(i=0;i<l;i++)
            if(nonzero[i])
            {
                model->SV[p] = (struct svm_node *)(perm[i]);
                model->sv_indices[p++] = perm[i] + 1;
            }
        
        int *nz_start = (int *)malloc(sizeof(int) * nr_class);
        nz_start[0] = 0;
        for(i=1;i<nr_class;i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];

        
        p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];
                int sj = start[j];
                int ci = count[i];
                int cj = count[j];
                
                int q = nz_start[i];
                int k;
                for(k=0;k<ci;k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];
                q = nz_start[j];
                for(k=0;k<cj;k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];
                ++p;
            }
        
        free(label);
        free(probA);
        free(probB);
        free(count);
        free(perm);
        free(start);
        free(x);
        free(weighted_C);
        free(nonzero);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            free(f[i].alpha);
        free(f);
        free(nz_count);
        free(nz_start);
    }
}



__global__ void cuda_svm_train_kernel(const struct svm_problem *subprobs, const struct svm_parameter *params, struct svm_model *submodels, size_t pitch, int nr_grid, int nr_fold)
{
    int x = blockIdx.x;
    int y = threadIdx.x;
    
    if (x<nr_grid && y<nr_fold)
    {
        struct svm_model *row = (struct svm_model *)((char*)submodels + x * pitch);
        cuda_perform_svm_train(&(subprobs[y]), &(params[x]), &(row[y]));
    }
}





int cuda_svm_train(const struct svm_problem *h_prob, struct svm_problem *h_subprobs, struct svm_parameter *h_params, int nr_grid, int nr_fold, struct svm_model *h_submodels)
{
    int i, j, k;
    int dev_cnt;
    int res = 0;
    
    //
    // Initialize
    //
    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "Error when initialize CUDA device\n");
        return 1;
    }
    
    cudaGetDeviceCount(&dev_cnt);
    if (dev_cnt == 0)
    {
         fprintf(stderr, "No CUDA device\n");
         return 1;
    }
    if (dev_cnt > 1)
    {
        //
        // Choose device that has maximum device memory left
        //
        int dev;
        int max_dev;
        int max_avail = 0;
        for (dev=0; dev<dev_cnt; dev++)
        {
            size_t avail;
            size_t total;
            cudaSetDevice(dev);
            cudaMemGetInfo(&avail, &total);
            if (avail > max_avail)
            {
                max_dev = dev;
                max_avail = avail;
            }
        }
        cudaSetDevice(max_dev);
    }
    
    struct svm_node **x_space = (struct svm_node **)malloc(sizeof(struct svm_node *) * h_prob->l);
    
    struct svm_problem *subprobs;
    struct svm_parameter *params;
    struct svm_model *submodels;
    
    // Send original prob.x to device
    for (i=0; i<h_prob->l; i++)
    {
        j=0;
        while(h_prob->x[i][j++].index != -1);
        cudaMalloc(&(x_space[i]), sizeof(struct svm_node) * j);
        cudaMemcpy(x_space[i], h_prob->x[i], sizeof(struct svm_node) * j, cudaMemcpyHostToDevice);
    }
    
    // Build subprobs in device
    cudaMalloc(&subprobs,  sizeof(struct svm_problem) * nr_fold);
    float **y = (float **)malloc(sizeof(float *) * nr_fold);
    struct svm_node ***x = (struct svm_node ***)malloc(sizeof(struct svm_node **) * nr_fold);
    for (i=0; i<nr_fold; i++)
    {
        cudaMemcpy(&(subprobs[i].l), &(h_subprobs[i].l), sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&(y[i]), sizeof(float) * h_subprobs[i].l);
        cudaMemcpy(y[i], h_subprobs[i].y, sizeof(float) * h_subprobs[i].l, cudaMemcpyHostToDevice);
        cudaMemcpy(&(subprobs[i].y), &(y[i]), sizeof(float *), cudaMemcpyHostToDevice);
        cudaMalloc(&(x[i]), sizeof(struct svm_node *) * h_subprobs[i].l);
        // The h_subprobs[i].x[j] stores the index in the original prob.x, instead of a pointer to that
        for (j=0; j<h_subprobs[i].l; j++)
            cudaMemcpy(&(x[i][j]), &(x_space[int(h_subprobs[i].x[j])]), sizeof(struct svm_node *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(subprobs[i].x), &(x[i]), sizeof(struct svm_node **), cudaMemcpyHostToDevice);
    }
    
    // Send params to device
    cudaMalloc(&params,    sizeof(struct svm_parameter) * nr_grid);
    cudaMemcpy(params, h_params, sizeof(struct svm_parameter) * nr_grid, cudaMemcpyHostToDevice);
    
    // Build results (submodels) cache in device
    size_t pitch;
    cudaMallocPitch(&submodels, &pitch, sizeof(struct svm_model) * nr_fold,  nr_grid);
    int nr_class = svm_get_nr_classes(h_prob);
    
    struct svm_node ***SV = (struct svm_node ***)malloc(sizeof(struct svm_node **) * nr_grid * nr_fold);
    
    // The model->sv_coef is a two-dimension linked list
    float ***sv_coef_p = (float ***)malloc(sizeof(float **) * nr_grid * nr_fold);
    float ***sv_coef   = (float ***)malloc(sizeof(float **) * nr_grid * nr_fold);
    for (i=0; i<nr_grid; i++)
        for (j=0; j<nr_fold; j++)
            sv_coef[i*nr_fold+j] = (float **)malloc(sizeof(float *) * (nr_class-1));
    
    float **rho        = (float **)malloc(sizeof(float *) * nr_grid * nr_fold);
    float **probA      = (float **)malloc(sizeof(float *) * nr_grid * nr_fold);
    float **probB      = (float **)malloc(sizeof(float *) * nr_grid * nr_fold);
    int    **sv_indices = (int **)malloc(sizeof(int *) * nr_grid * nr_fold);
    int    **label      = (int **)malloc(sizeof(int *) * nr_grid * nr_fold);
    int    **nSV        = (int **)malloc(sizeof(int *) * nr_grid * nr_fold);
    
    // Build the storage structure for results
    for (i=0; i<nr_grid; i++)
        for (j=0; j<nr_fold; j++)
        {
            cudaMalloc(&(SV[i*nr_fold+j]), sizeof(struct svm_node *) * h_subprobs[j].l);
            
            cudaMalloc(&(sv_coef_p[i*nr_fold+j]), sizeof(float *) * (nr_class-1));
            for (k=0; k<nr_class-1; k++)
                cudaMalloc(&(sv_coef[i*nr_fold+j][k]), sizeof(float) * h_subprobs[j].l);
            
            cudaMalloc(&(rho[i*nr_fold+j]),   sizeof(float) * nr_class*(nr_class-1)/2);
            cudaMalloc(&(probA[i*nr_fold+j]), sizeof(float) * nr_class*(nr_class-1)/2);
            cudaMalloc(&(probB[i*nr_fold+j]), sizeof(float) * nr_class*(nr_class-1)/2);
            cudaMalloc(&(sv_indices[i*nr_fold+j]), sizeof(int) * h_subprobs[j].l);
            cudaMalloc(&(label[i*nr_fold+j]), sizeof(int) * nr_class);
            cudaMalloc(&(nSV[i*nr_fold+j]),   sizeof(int) * nr_class);
            
            struct svm_model *models = (struct svm_model *)((char*)submodels + i*pitch);
            
            cudaMemcpy(&(models[j].SV),    &(SV[i*nr_fold+j]),   sizeof(struct svm_node **), cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(models[j].sv_coef), &(sv_coef_p[i*nr_fold+j]), sizeof(float **), cudaMemcpyHostToDevice);
            cudaMemcpy(sv_coef_p[i*nr_fold+j], sv_coef[i*nr_fold+j], sizeof(float *) * (nr_class-1), cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(models[j].rho),   &(rho[i*nr_fold+j]),   sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(models[j].probA), &(probA[i*nr_fold+j]), sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(models[j].probB), &(probB[i*nr_fold+j]), sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(models[j].sv_indices), &(sv_indices[i*nr_fold+j]), sizeof(int *),  cudaMemcpyHostToDevice);
            cudaMemcpy(&(models[j].label), &(label[i*nr_fold+j]), sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(models[j].nSV),   &(nSV[i*nr_fold+j]),   sizeof(int *), cudaMemcpyHostToDevice);
        }
    
    
    //
    // Run the kernel
    //
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_HEAP_SIZE);
    cuda_svm_train_kernel<<<nr_grid, nr_fold>>>(subprobs, params, submodels, pitch, nr_grid, nr_fold);
    
    
    if (cudaGetLastError() == cudaSuccess)
    {
        if (cudaDeviceSynchronize() == cudaSuccess)
        {
            // Copy results (submodels) from device to host
            cudaMemcpy2D(h_submodels, sizeof(struct svm_model) * nr_fold, submodels, pitch, sizeof(struct svm_model) * nr_fold, nr_grid, cudaMemcpyDeviceToHost);
            for (i=0; i<nr_grid; i++)
                for (j=0; j<nr_fold; j++)
                {
                    struct svm_model *model = h_submodels + i*nr_fold + j;
                    
                    model->SV         = (struct svm_node **)malloc(sizeof(struct svm_node *) * h_subprobs[j].l);
                    cudaMemcpy(model->SV, SV[i*nr_fold+j], sizeof(struct svm_node *) * h_subprobs[j].l, cudaMemcpyDeviceToHost);
                    
                    // The returned model->SV is a index to the subprob, instead of a pointer
                    for(k=0; k<model->l; k++)
                        model->SV[k] = (struct svm_node *)(h_prob->x[int(h_subprobs[j].x[int(model->SV[k])])]);
                    
                    model->sv_coef    = (float **)malloc(sizeof(float *) * (nr_class-1));
                    for (k=0; k<nr_class-1; k++)
                    {
                        model->sv_coef[k] = (float *)malloc(sizeof(float) * h_subprobs[j].l);
                        cudaMemcpy(model->sv_coef[k], sv_coef[i*nr_fold+j][k], sizeof(float) * h_subprobs[j].l, cudaMemcpyDeviceToHost);
                    }
                    
                    model->rho        = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
                    model->probA      = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
                    model->probB      = (float *)malloc(sizeof(float) * nr_class*(nr_class-1)/2);
                    model->sv_indices = (int *)malloc(sizeof(int) * h_subprobs[j].l);
                    model->label      = (int *)malloc(sizeof(int) * nr_class);
                    model->nSV        = (int *)malloc(sizeof(int) * nr_class);
                    
                    cudaMemcpy(model->rho,        rho[i*nr_fold+j],   sizeof(float) * nr_class*(nr_class-1)/2, cudaMemcpyDeviceToHost);
                    cudaMemcpy(model->probA,      probA[i*nr_fold+j], sizeof(float) * nr_class*(nr_class-1)/2, cudaMemcpyDeviceToHost);
                    cudaMemcpy(model->probB,      probB[i*nr_fold+j], sizeof(float) * nr_class*(nr_class-1)/2, cudaMemcpyDeviceToHost);
                    cudaMemcpy(model->sv_indices, sv_indices[i*nr_fold+j], sizeof(int) * h_subprobs[j].l,       cudaMemcpyDeviceToHost);
                    cudaMemcpy(model->label,      label[i*nr_fold+j], sizeof(int) * nr_class, cudaMemcpyDeviceToHost);
                    cudaMemcpy(model->nSV,        nSV[i*nr_fold+j],   sizeof(int) * nr_class, cudaMemcpyDeviceToHost);
                }
        }
        else
        {
            fprintf(stderr, "Error when running CUDA svm train: %s\n", cudaGetErrorString(cudaGetLastError()));
            res = 1;
        }
    }
    else
    {
        fprintf(stderr, "Error when launching CUDA svm train\n");
        res = 1;
    }
    
    
    //
    // Free all the memory allocated in device and host
    //
    for (i=0; i<h_prob->l; i++)
        cudaFree(x_space[i]);
    free(x_space);
    
    for (i=0; i<nr_fold; i++)
    {
        cudaFree(y[i]);
        cudaFree(x[i]);
    }
    free(y);
    free(x);
    cudaFree(subprobs);
    
    cudaFree(params);
    cudaFree(submodels);
    
    for (i=0; i<nr_grid; i++)
        for (j=0; j<nr_fold; j++)
        {
            cudaFree(SV[i*nr_fold+j]);
            
            for (k=0; k<nr_class-1; k++)
                cudaFree(sv_coef[i*nr_fold+j][k]);
            free(sv_coef[i*nr_fold+j]);
            cudaFree(sv_coef_p[i*nr_fold+j]);
            
            cudaFree(rho[i*nr_fold+j]);
            cudaFree(probA[i*nr_fold+j]);
            cudaFree(probB[i*nr_fold+j]);
            cudaFree(sv_indices[i*nr_fold+j]);
            cudaFree(label[i*nr_fold+j]);
            cudaFree(nSV[i*nr_fold+j]);
        }
    free(SV);
    free(sv_coef_p);
    free(sv_coef);
    free(rho);
    free(probA);
    free(probB);
    free(sv_indices);
    free(label);
    free(nSV);
    
    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "Error when cleaning CUDA svm train\n");
        res = 1;
    }
    
    return res;
}



