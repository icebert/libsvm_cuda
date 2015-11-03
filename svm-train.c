#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
    printf(
    "Usage: svm-train [options] training_set_file [model_file]\n"
    "options:\n"
    "-s svm_type : set type of SVM (default 0)\n"
    "    0 -- C-SVC        (multi-class classification)\n"
    "    1 -- nu-SVC        (multi-class classification)\n"
    "    2 -- one-class SVM\n"
    "    3 -- epsilon-SVR    (regression)\n"
    "    4 -- nu-SVR        (regression)\n"
    "-t kernel_type : set type of kernel function (default 2)\n"
    "    0 -- linear: u'*v\n"
    "    1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
    "    2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
    "    3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
    "    4 -- precomputed kernel (kernel values in training_set_file)\n"
    "-d degree : set degree in kernel function (default 3)\n"
    "-g gamma : set gamma in kernel function (default 1/num_features)\n"
    "-r coef0 : set coef0 in kernel function (default 0)\n"
    "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
    "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
    "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
    "-m cachesize : set cache memory size in MB (default 20)\n"
    "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
    "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
    "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
    "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
    "-v n: n-fold cross validation mode\n"
    "-C start,end,step : find best cost\n"
    "-G start,end,step : find best gamma\n"
    "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();
void find_param_c_g();

struct svm_parameter param;        // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int find_c;
float c_begin_p,c_end_p,c_step_p;
int find_g;
float g_begin_p,g_end_p,g_step_p;
int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;
    
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

int main(int argc, char **argv)
{
    char input_file_name[1024];
    char model_file_name[1024];
    const char *error_msg;

    parse_command_line(argc, argv, input_file_name, model_file_name);
    read_problem(input_file_name);
    error_msg = svm_check_parameter(&prob,&param);

    if(error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }

    if (find_c || find_g)
    {
        find_param_c_g();
    }
    else if(cross_validation)
    {
        do_cross_validation();
    }
    else
    {
        model = svm_train(&prob,&param);
        if(svm_save_model(model_file_name,model))
        {
            fprintf(stderr, "can't save model to file %s\n", model_file_name);
            exit(1);
        }
        svm_free_and_destroy_model(&model);
    }
    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(x_space);
    free(line);

    return 0;
}

void do_cross_validation()
{
    int i;
    int total_correct = 0;
    float total_error = 0;
    float sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    float *target = Malloc(float,prob.l);

    svm_cross_validation(&prob,&param,nr_fold,target);
    if(param.svm_type == EPSILON_SVR ||
       param.svm_type == NU_SVR)
    {
        for(i=0;i<prob.l;i++)
        {
            float y = prob.y[i];
            float v = target[i];
            total_error += (v-y)*(v-y);
            sumv += v;
            sumy += y;
            sumvv += v*v;
            sumyy += y*y;
            sumvy += v*y;
        }
        printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
        printf("Cross Validation Squared correlation coefficient = %g\n",
            ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
            ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
            );
    }
    else
    {
        for(i=0;i<prob.l;i++)
            if(target[i] == prob.y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
    }
    free(target);
}



void permute_sequence(float *seq, int left, int right, float *res)
{
    if (right - left < 1)
        return;
    if (right - left == 1)
    {
        res[0] = seq[left];
        return;
    }
    
    int mid = (left + right) / 2;
    int llen = mid - left;
    int rlen = right - mid -1;
    
    float *left_seq = NULL;
    float *right_seq = NULL;
    if (llen > 0) left_seq = Malloc(float, llen);
    if (rlen > 0) right_seq = Malloc(float, rlen);
    
    permute_sequence(seq, left, mid, left_seq);
    permute_sequence(seq, mid+1, right, right_seq);
    
    int cnt=0, i=0, j=0;
    res[cnt++] = seq[mid];
    while(i<llen || j<rlen)
    {
        if (i<llen) res[cnt++] = left_seq[i++];
        if (j<rlen) res[cnt++] = right_seq[j++];
    }
    
    if (left_seq != NULL) free(left_seq);
    if (right_seq != NULL) free(right_seq);
}

void find_param_c_g()
{
    struct grid_cg grid;
    float *target;
    
    float c_begin = param.C;
    float c_end = param.C;
    float c_step = 1;
    float c;
    float *c_seq, *c_seq_p;
    int c_len = 0;
    
    float g_begin = param.gamma;
    float g_end = param.gamma;
    float g_step = -1;
    float g;
    float *g_seq, *g_seq_p;
    int g_len = 0;
    
    if (find_c)
    {
        if (c_begin_p < c_end_p && c_step_p > 0)
        {
            c_begin = c_begin_p;
            c_end   = c_end_p;
            c_step  = c_step_p;
        }
        if (c_begin_p > c_end_p && c_step_p < 0)
        {
            c_begin = c_end_p;
            c_end   = c_begin_p;
            c_step  = -c_step_p;
        }
    }
    
    if (find_g)
    {
        if (g_begin_p < g_end_p && g_step_p > 0)
        {
            g_begin = g_begin_p;
            g_end   = g_end_p;
            g_step  = g_step_p;
        }
        if (g_begin_p > g_end_p && g_step_p < 0)
        {
            g_begin = g_end_p;
            g_end   = g_begin_p;
            g_step  = -g_step_p;
        }
    }
    
    c_seq   = Malloc(float, int((c_end-c_begin)/c_step)+1);
    c_seq_p = Malloc(float, int((c_end-c_begin)/c_step)+1);
    g_seq   = Malloc(float, int((g_end-g_begin)/g_step)+1);
    g_seq_p = Malloc(float, int((g_end-g_begin)/g_step)+1);
    
    for (c=c_begin; c<=c_end; c+=c_step)
        c_seq[c_len++] = c;
    for (g=g_begin; g>=g_end; g+=g_step)
        g_seq[g_len++] = g;
    
    permute_sequence(c_seq, 0, c_len, c_seq_p);
    permute_sequence(g_seq, 0, g_len, g_seq_p);
    
    grid.l = 0;
    grid.c = Malloc(float, c_len*g_len);
    grid.g = Malloc(float, c_len*g_len);
    
    int i=0, j=0, k;
    while (i < c_len || j < g_len)
        if (1.0*i/c_len < 1.0*j/g_len)
        {
            // Increase c resolution
            for (k=0; k<j; k++)
            {
                grid.c[grid.l] = c_seq_p[i];
                grid.g[grid.l] = g_seq_p[k];
                grid.l++;
            }
            i++;
        }
        else
        {
            // Increase g resolution
            for (k=0; k<i; k++)
            {
                grid.c[grid.l] = c_seq_p[k];
                grid.g[grid.l] = g_seq_p[j];
                grid.l++;
            }
            j++;
        }
    
    for (k=0; k<grid.l; k++)
    {
        if (find_c) grid.c[k] = powf(2, grid.c[k]);
        if (find_g) grid.g[k] = powf(2, grid.g[k]);
    }
    
    if (nr_fold == 0) nr_fold = 5;
    target = Malloc(float, grid.l * prob.l);
    
    int res = svm_grid_search(&prob, &param, &grid, nr_fold, target);
    
    if (res == 0)
    {
        float accu;
        float best_accu = 0;
        float best_c=param.C, best_g=param.gamma;
        for (k=0; k<grid.l; k++)
        {
            if (find_c) printf("%f\t", log2(grid.c[k]));
            if (find_g) printf("%f\t", log2(grid.g[k]));
            
            if(param.svm_type == EPSILON_SVR ||
               param.svm_type == NU_SVR)
            {
                float total_error = 0;
                for(i=0;i<prob.l;i++)
                {
                    float y = prob.y[i];
                    float v = target[k*prob.l + i];
                    total_error += (v-y)*(v-y);
                }
                accu = total_error/prob.l;
            }
            else
            {
                int total_correct = 0;
                for(i=0;i<prob.l;i++)
                    if(target[k*prob.l + i] == prob.y[i])
                        ++total_correct;
                accu = 100.0*total_correct/prob.l;
            }
            
            printf("%f\n", accu);
            
            if (accu > best_accu)
            {
                best_accu = accu;
                best_c = grid.c[k];
                best_g = grid.g[k];
            }
        }
        printf("Best\t");
        if (find_c) printf("cost: %f\t", best_c);
        if (find_g) printf("gamma: %f\t", best_g);
        printf("accuracy: %f\n", best_accu);
    }
    
    free(c_seq);
    free(c_seq_p);
    free(g_seq);
    free(g_seq_p);
    free(grid.c);
    free(grid.g);
    free(target);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i;
    void (*print_func)(const char*) = NULL;    // default printing to stdout

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;    // 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 20;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    cross_validation = 0;

    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc)
            exit_with_help();
        switch(argv[i-1][1])
        {
            case 's':
                param.svm_type = atoi(argv[i]);
                break;
            case 't':
                param.kernel_type = atoi(argv[i]);
                break;
            case 'd':
                param.degree = atoi(argv[i]);
                break;
            case 'g':
                param.gamma = atof(argv[i]);
                break;
            case 'r':
                param.coef0 = atof(argv[i]);
                break;
            case 'n':
                param.nu = atof(argv[i]);
                break;
            case 'm':
                param.cache_size = atof(argv[i]);
                break;
            case 'c':
                param.C = atof(argv[i]);
                break;
            case 'e':
                param.eps = atof(argv[i]);
                break;
            case 'p':
                param.p = atof(argv[i]);
                break;
            case 'h':
                param.shrinking = atoi(argv[i]);
                break;
            case 'b':
                param.probability = atoi(argv[i]);
                break;
            case 'q':
                print_func = &print_null;
                i--;
                break;
            case 'v':
                cross_validation = 1;
                nr_fold = atoi(argv[i]);
                if(nr_fold < 2)
                {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    exit_with_help();
                }
                break;
            case 'C':
                find_c = 1;
                sscanf(argv[i], "%f,%f,%f", &c_begin_p, &c_end_p, &c_step_p);
                break;
            case 'G':
                find_g = 1;
                sscanf(argv[i], "%f,%f,%f", &g_begin_p, &g_end_p, &g_step_p);
                break;
            case 'w':
                ++param.nr_weight;
                param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
                param.weight = (float *)realloc(param.weight,sizeof(float)*param.nr_weight);
                param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
                param.weight[param.nr_weight-1] = atof(argv[i]);
                break;
            default:
                fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
        }
    }

    svm_set_print_string_function(print_func);

    // determine filenames

    if(i>=argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if(i<argc-1)
        strcpy(model_file_name,argv[i+1]);
    else
    {
        char *p = strrchr(argv[i],'/');
        if(p==NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name,"%s.model",p);
    }
}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob.l = 0;
    elements = 0;

    max_line_len = 1024;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label

        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
        }
        ++elements;
        ++prob.l;
    }
    rewind(fp);

    prob.y = Malloc(float,prob.l);
    prob.x = Malloc(struct svm_node *,prob.l);
    x_space = Malloc(struct svm_node,elements);

    max_index = 0;
    j=0;
    for(i=0;i<prob.l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);

        prob.y[i] = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);

        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }

    if(param.gamma == 0 && max_index > 0)
        param.gamma = 1.0/max_index;

    if(param.kernel_type == PRECOMPUTED)
        for(i=0;i<prob.l;i++)
        {
            if (prob.x[i][0].index != 0)
            {
                fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                exit(1);
            }
            if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
            {
                fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                exit(1);
            }
        }

    fclose(fp);
}
