#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas.h>
#include <curand.h>


#include "read_files.h"
#include "distance.h"
#include "functions.h"
#include "kernels.h"


// ! ! ! ! ! ! ! ! ! ! ATTENTION ! ! ! ! ! ! ! ! ! !
// READ THE HANDOUT BEFORE TRYING TO RUN THIS PROGRAM
// ! ! ! ! ! ! ! ! ! ! ATTENTION ! ! ! ! ! ! ! ! ! !



// Some important variables  
int Burngamma = 150;  // How many iterations only for the update of long vectors \gamma
float epsilon = 0.00001f; // Initial value for the HMC stepsize
float tmp = 0.0f;
float Htau = 0.001f; // The prior precision for some of the parameters. Intended to make the normal prior uninformative. 
int Hd =3; // The dimension 
float voxel_volume = 8.00f; // The volume of each element of the grid. It is zero for elements outside the brain mask

int HV, HI, HN, HK_star, HK, L, Diff, V_extended;
long seed;

 
int main (int argc , char *argv[]) 
{
    printf("\n Neuroimaging CBMA via log-Gaussian Cox processes");
    printf("\n Beginning of the simulation");

    // Set the device that will run the simulation
    cudaSetDevice(1);

    /* Command line arguments */ 
    int Burnin = atoi(argv[1]);        // The burn-in period of the HMC         
    int Iterations = atoi(argv[2]);    // The total number of iterations AFTER burn-in
    int Adjust = atoi(argv[3]);        // How often to adjust the stepsize
    int Adjust_window = atoi(argv[4]); // Chain window when adjusting the stepsize
    int Thinning = atoi(argv[5]);      // How often to save the running sum of the GPs
    int Save = atoi(argv[6]);          // How often to save snapshots of the GPs


    int i, j, k, ii, kk;

    /* Some parameters of the dataset */
    FILE *file;
    file = fopen("./inputs/setup.txt","r");
    fscanf(file,"%d",&HV);       // total number of elements in the initial grid. The program will figure out how many there are in the extended
    fscanf(file,"%d",&HN);       // total number of points (foci)
    fscanf(file,"%d",&HI);       // total number of point patterns (contrasts/studies)
    fscanf(file,"%d",&HK_star);  // total number of covariates
    fscanf(file,"%d",&HK);       // total number of spatially varying covariates
    fscanf(file,"%d",&L);        // total number of HMC leapfrog steps
    fscanf(file,"%lu",&seed);    // seed
    float  tmp_mass[4];          // HMC mass parameters
    for (j=0 ; j<4 ; j++) { fscanf(file,"%f",&tmp_mass[j]); }
    fscanf(file,"%d",&Diff);     // If one wants to see between-type comparisons
    fclose(file);

    /* The device seed */
    unsigned long * RNG = (unsigned long *)calloc(3,sizeof(unsigned long));
    file = fopen("./inputs/seed.dat","r");
    if (fscanf(file,"%lu %lu %lu\n",&(RNG[0]),&(RNG[1]),&(RNG[2]))) {}
    fclose(file);

    /* If only one spatially varying intercept then there is no comparison to make */
    if (HK==0) {Diff=0;}

    /* Declare the GPU varibles here */
    cufftComplex *DFT , *DFT_big;
    cufftHandle  plan , plan_big;

    float  *sigma, *grad_sigma, *ori_sigma, *ori_grad_sigma, *mom_sigma, *mom_sigma_even;
    double *sigma_part;
    float *rho, *grad_rho, *ori_rho, *ori_grad_rho, *mom_rho, *mom_rho_even , *Sign_rho;
    double *rho_part;
    float *beta, *grad_beta, *ori_beta,  *ori_grad_beta,  *mom_beta,  *mom_beta_even;
    double *beta_part;
    float *gamma, *grad_gamma, *ori_gamma, *ori_grad_gamma, *mom_gamma, *mom_gamma_even; 

    float *Zc, *dist  , *eigen , *Cgamma  ,  *CSgamma , *ori_Cgamma , *ori_CSgamma;
    float *ZB , *vol , *voxel_tmp;
    float *COVARIATES , *TERM_GAMMA , *lik_tmp ;
    double *lik_part, *kin_part;
    
    double *likelihood, *Hamiltonian, *Kinetic; 
    double *gp_s, *gp_ss; /* running sums and sum of square of the gps */

    float *ub , *um , *Mass_sigma , *Mass_beta , *Mass_rho; 
    double ratio;
    unsigned int *d_mSteps ;
    int accepted;
    double *rfx;
    
    curandGenerator_t gen;
    cudaError_t error;
    
    char filename[32];
    FILE *FIELDS;

   
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Memory allocations 

    // This matrix has the coordinates of the points in voxel space. The last column is the point pattern (contrast) identifier. 
    int **Hfoci = (int **)malloc(sizeof(int*)*HN);
    for (i=0 ; i<HN ; i++) {
        Hfoci[i] = (int *)malloc(sizeof(int)*(Hd+1));
    }
    // Z is the design matrix. The first column always contains 1. The next columns are the spatially varying coefficients.
    float **HZ = (float **)malloc(sizeof(float*)*HI);
    for (i=0 ; i<HI ; i++) {
        HZ[i] = (float *)malloc(sizeof(float)*HK_star);
    }
    // Hcounts is the number of points per point pattern (contrast). It is usefull when calculating the likelihood.
    int *Hcounts = (int*) malloc(HI*sizeof(int));
    /* Paper index */
    int *author = (int*) malloc(HI*sizeof(int));
    // The 0/1 brain mask 
    int *brain = (int *)malloc(sizeof(int)*HV);
    read_files(Hfoci,HZ,Hcounts,brain,author);
    
    // This bit vectorizes the design matrix into column major order. We need it that way to do multiplications with CUBLAS.
    int tmp_i;
    float *HZc =    (float*) malloc(sizeof(float)*HI*HK_star);
    for (i=0 ; i<HK_star ; i++) {
        for (j=0 ; j<HI ; j++){
            tmp_i = j+i*HI;
            HZc[tmp_i] = HZ[j][i];
        }
    }
    
    // Transfers the data to the device
    cudaMalloc((void**)&Zc,sizeof(float)*HI*HK_star);
    cudaMemcpy(Zc,HZc,sizeof(float)*HI*HK_star,cudaMemcpyHostToDevice); // Now we don't need HZc in the CPU anymore. I free the memory at the end though.
    

    // The following vector is essential for gradient calculations. Row major design matrix. One extra term is reserved per study, the exponential ZB term.
    // It's length needs to be a multiple of the number of threads used by the gradient kernel. A small part of it at the end is not used. 
    float *HCOVARIATES = (float*) calloc(TIMES*NT,sizeof(float));
    cudaMalloc((void**)&COVARIATES,sizeof(float)*TIMES*NT);
    for (i=0 ; i<HI ; i++){
        HCOVARIATES[i*(HK_star+2)] = 0.0f; /* RFX CHANGE */
        for (j=0 ; j<HK_star ; j++) {
            HCOVARIATES[i*(HK_star+2)+1+j] = HZ[i][j];
        }
        HCOVARIATES[i*(HK_star+2)+1+HK_star] = 1.0f; /* RFX CHANGE */
    }
    cudaMemcpy(COVARIATES,HCOVARIATES,sizeof(float)*TIMES*NT,cudaMemcpyHostToDevice);

    printf("\nSTEP 1: The point pattern data have been read and transferred to the GPU!");
    fflush(NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Set up the grid 
    // if(Hd==3): so it is easier to extend to the 2D case 

    int *Hgrid = (int*)calloc(Hd,sizeof(int)); // The size of the original grid
    int *grid_extended = (int*)calloc(Hd,sizeof(int)); // The minimum extension in order for the FFT to work
    // Depending on the problem set the following values as you wish
    if (Hd==3) {
        Hgrid[0] = 69;
        Hgrid[1] = 88;
        Hgrid[2] = 70; // Those are the dimensions of our 2x2x2 MNI brain atlas.
    }
    
    // The following three lines define the extended grid
    if (Hd==3) {
        grid_extended[0] = 144;
        grid_extended[1] = 192;
        grid_extended[2] = 144;
        V_extended = grid_extended[0]*grid_extended[1]*grid_extended[2];
    }

    // CAREFUL: The total number of voxels in the extended grid must be manually set in the Kernel and Function files. 
    int *brain_extended = (int*)malloc(V_extended*sizeof(int)); // this is the new binary mask
    // Initialise everything to zero
    for (i=0 ; i<V_extended ; i++) {
        brain_extended[i] = 0;
    }
    // Then replace the elements from the original mask
    if (Hd==3){
        for (k=0;k<Hgrid[2];k++){
            for (j=0;j<Hgrid[1];j++) {
                for (i=0;i<Hgrid[0];i++) {
                    brain_extended[ i + j*grid_extended[0] + k*grid_extended[0]*grid_extended[1] ] = brain[ i + j*Hgrid[0] + k*Hgrid[0]*Hgrid[1] ];
                }
            }
        }
    }

    // ID says where each focus lies
    int *HID = (int*) malloc(HN*sizeof(int*));
    if( Hd==3 ) {
        for ( i=0 ; i<HN ; i++ ) {
            HID[i] = (Hfoci[i][0]-1)+(Hfoci[i][1]-1)*grid_extended[0]+(Hfoci[i][2]-1)*grid_extended[0]*grid_extended[1];
        }
    }
    
    // Distances of the base elements
    float *Hdist = (float *) calloc(V_extended,sizeof(float)); 
    distance( Hdist , Hd , grid_extended , Hgrid);
    cudaMalloc((void**)&dist,sizeof(float)*V_extended);
    cudaMemcpy(dist,Hdist,sizeof(float)*V_extended,cudaMemcpyHostToDevice);
    
    printf("\nSTEP 2: The new grid has been set.");
    fflush(NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PART 3: Set up the FFT staff
    // READ THIS. FFT is used to multiply Nested Block Circulant matrices with the \gamma vectors. One NBC matrix is the square root of the correlation matrix and the other 
    // one is the matrix in the gradient of \rho. Since its faster to calculate their eigenvalues with FFT batch mode we only create one object for both.
    // However, since the vector that these are multiplied is the same every time, we make a small object to find the IDFT of the \gamma vectors only once.  
   
    cudaMalloc((void**)&DFT     ,sizeof(cufftComplex)*V_extended*HK    );
    cudaMalloc((void**)&DFT_big ,sizeof(cufftComplex)*V_extended*HK*2  );
 
    // now the plan can be created. instead of doing each one seperately, use the batching mode
    if ( Hd==3 ) {
        int plan_dim[3] = { grid_extended[0] , grid_extended[1] , grid_extended[2] };
        cufftPlanMany(&plan, 3, plan_dim, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, HK  );
        cufftPlanMany(&plan_big, 3, plan_dim, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 2*HK  );
    }
    
    printf("\nSTEP 3: FFT plans have been initialised. The following errors have occured:");
    error = cudaGetLastError();
    printf("\n  CUDA error: %s\n", cudaGetErrorString(error));
    fflush(NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PART 4: Read starting values and allocate model parameters

    cudaMalloc((void**)&eigen,sizeof(float)*V_extended*HK);      // Contains the square root eigenvalues of the correlation matrices. NOT OF THE SECOND MATRIX!!!!
    cudaMalloc((void**)&Cgamma,sizeof(float)*V_extended*HK);     // Contains the product C\gamma
    cudaMalloc((void**)&CSgamma,sizeof(float)*V_extended*HK);    // For the gradient of \rho

    // Define and read the correlation decay parameters.
    cudaMalloc((void**)&rho,sizeof(float)*HK);               
    float *Hrho = (float*)malloc(HK*sizeof(float));                
    file = fopen("./inputs/rho.txt","r");
    for (i=0 ; i<HK ; i++) {
        if( !fscanf(file,"%f",&Hrho[i]) )
            break;
    }
    fclose(file);
    cudaMemcpy(rho,Hrho,sizeof(float)*HK,cudaMemcpyHostToDevice);

    // Marginal standard deviations
    cudaMalloc((void**)&sigma,sizeof(float)*HK);               
    float *Hsigma = (float*)malloc(HK*sizeof(float));               
    file = fopen("./inputs/sigma.txt","r");
    for (i=0 ; i<HK ; i++) {
        if( !fscanf(file,"%f",&Hsigma[i]) )
            break;
    }
    fclose(file);
    cudaMemcpy(sigma,Hsigma,sizeof(float)*HK,cudaMemcpyHostToDevice);

    // Overall mean parameters
    cudaMalloc((void**)&beta,sizeof(float)*HK_star);  
    float *Hbeta = (float*)malloc(HK_star*sizeof(float));            
    file = fopen("./inputs/beta.txt","r");
    for (i=0 ; i<HK_star ; i++) {
        if( !fscanf(file,"%f",&Hbeta[i]) )
            break;
    }
    fclose(file);
    cudaMemcpy(beta,Hbeta,sizeof(float)*HK_star,cudaMemcpyHostToDevice);

    // Standard normal vectors
    cudaMalloc((void**)&gamma,sizeof(float)*V_extended*HK);    
    float *Hgamma = (float*)malloc(V_extended*HK*sizeof(float));    
    file = fopen("./inputs/gamma.txt","r");
    for (i=0 ; i<(HK*V_extended) ; i++) {
        if( !fscanf(file,"%f",&Hgamma[i]) )
            break;
    }
    fclose(file);
    cudaMemcpy(gamma,Hgamma,sizeof(float)*HK*V_extended,cudaMemcpyHostToDevice);

    /* RFX CHANGE */
    cudaMalloc((void**)&rfx,sizeof(double)*HI);   // the intensity random effects 
    double* Hrfx = (double*) calloc(HI,sizeof(double));
    for (i=0 ; i<HI ; i++) {
        Hrfx[i] = 1.0;
    }
    cudaMemcpy(rfx,Hrfx,sizeof(double)*HI,cudaMemcpyHostToDevice);
    /* RFX CHANGE */
    printf("\nSTEP 4: Initial values have been loaded.");
    fflush(NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PART 5: Gradient and likelihood
    
    // The product of the design matrix with the vector of per pattern counts remains constant throughout the simulation but is used a lot. We find it here.
    float *HZN = (float*)calloc(HK_star,sizeof(float));
    for (i=0 ; i<HK_star ; i++) {
        for (j=0 ; j<HI ; j++) {
            HZN[i] += Hcounts[j]*HZ[j][i];
        }
    }
    
    // The product of the design matrix with \beta is constantly changing. A kernel does the multiplication in CUBLAS.
    float *HZB = (float*)calloc(HI,sizeof(float));
    cudaMalloc((void**)&ZB,HI*sizeof(float));
    cudaMemcpy(ZB,HZB,HI*sizeof(float),cudaMemcpyHostToDevice);
    cublasInit();
    
    // The vector volume is the Lebesgue measure of the elements in our grid. Zero values indicate points that are not considered in the likelihood. 
    float *Hvol = (float*)calloc(V_extended,sizeof(float));
    for (i=0 ; i<V_extended ; i++) {
        if (brain_extended[i]==1) {
            Hvol[i] = voxel_volume;
        } else {
            Hvol[i] = 0.0f;
        }

    }
    cudaMalloc((void**)&vol,V_extended*sizeof(float));
    cudaMemcpy(vol,Hvol,V_extended*sizeof(float),cudaMemcpyHostToDevice);


    // Gradient variables are defined in the GPU
    cudaMalloc((void**)& grad_gamma,HK*V_extended*sizeof(float));
    cudaMalloc((void**)& grad_sigma,HK*sizeof(float));
    cudaMalloc((void**)& grad_rho  ,HK*sizeof(float));
    cudaMalloc((void**)&grad_beta,HK_star*sizeof(float));

    // Useful for gradient calculations. No meaning. 
    cudaMalloc((void**)&voxel_tmp,HK_star*V_extended*sizeof(float));
    
    // Some terms in the gradient do not change with simulation time. We calculate once and use within kernels. 
    float *HTERM_GAMMA = (float*) calloc(HK*V_extended,sizeof(float));
    for (i=0 ; i<HK*V_extended ; i++){
        HTERM_GAMMA[i] = 0.0f;
    }
    int idx_tmp1 , idx_tmp2;
    for (i=0 ; i<HN ; i++) {
        idx_tmp1 = HID[i];
        idx_tmp2 = Hfoci[i][Hd]-1;
        for (j=0 ; j<HK ; j++) {
            HTERM_GAMMA[idx_tmp1+j*V_extended] += HZ[idx_tmp2][j];
        }
    }
    cudaMalloc((void**)&TERM_GAMMA,HK*V_extended*sizeof(float));
    cudaMemcpy(TERM_GAMMA,HTERM_GAMMA,HK*V_extended*sizeof(float),cudaMemcpyHostToDevice);
    

    // Partial sums for some gradient terms. 
    cudaMalloc((void**)&sigma_part,NB*HK*sizeof(double));
    cudaMalloc((void**)&rho_part,  NB*HK*sizeof(double));
    cudaMalloc((void**)&beta_part,NB*HK_star*sizeof(double));
    double *Hsigma_part = (double*) calloc(NB*HK,sizeof(double));
    double *Hrho_part   = (double*) calloc(NB*HK,sizeof(double));
    double *Hbeta_part  = (double*) calloc(NB*HK_star,sizeof(double));
    
    cudaMalloc( (void**)&lik_tmp , V_extended*sizeof(float) );      // This is a temporary variable to save voxel values for the likelihood
    cudaMalloc( (void**)&lik_part , NB*sizeof(double) );            // This holds the likelihood partial sums in the GPU
    double *Hlik_part = (double*) malloc(NB*sizeof(double));                // This holds the likelihood partial sums in the host
    likelihood = (double*) calloc(2,sizeof(double));                // This holds the values of the likelihood: 0 = proposed , 1 = original

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,seed);
    
    error = cudaGetLastError();
    printf("\n  Random number generator error: %s\n", cudaGetErrorString(error));
    
    // The original values at the beggining of each iteration. Potentially replaced with the proposed values
    cudaMalloc( (void**)&ori_beta   , HK_star*sizeof(float) );
    cudaMalloc( (void**)&ori_sigma  , HK*sizeof(float) );
    cudaMalloc( (void**)&ori_rho    , HK*sizeof(float) );
    cudaMalloc( (void**)&ori_gamma  , HK*V_extended*sizeof(float) );
    cudaMalloc( (void**)&ori_Cgamma  , HK*V_extended*sizeof(float) );
    cudaMalloc( (void**)&ori_CSgamma  , HK*V_extended*sizeof(float) );
    
    cudaMalloc( (void**)&ori_grad_beta   , HK_star*sizeof(float) );
    cudaMalloc( (void**)&ori_grad_sigma  , HK*sizeof(float) );
    cudaMalloc( (void**)&ori_grad_rho    , HK*sizeof(float) );
    cudaMalloc( (void**)&ori_grad_gamma  , HK*V_extended*sizeof(float) );

    cudaMalloc( (void**)&mom_beta   , HK_star*sizeof(float) );
    cudaMalloc( (void**)&mom_sigma  , HK*sizeof(float) );
    cudaMalloc( (void**)&mom_rho    , HK*sizeof(float) );
    cudaMalloc( (void**)&mom_gamma  , HK*V_extended*sizeof(float) );
    float *Hmom_beta  = (float*) malloc(HK_star*sizeof(float));
    float *Hmom_sigma = (float*) malloc(HK*sizeof(float));
    float *Hmom_rho   = (float*) malloc(HK*sizeof(float));
    
    // The generator cannot produce even number of values. Whenever this is the need, we just make an extra one which we omit.
    cudaMalloc( (void**)&mom_beta_even   , (HK_star+1)*sizeof(float) );
    cudaMalloc( (void**)&mom_sigma_even  , (HK+1)*sizeof(float) );
    cudaMalloc( (void**)&mom_rho_even    , (HK+1)*sizeof(float) );
    cudaMalloc( (void**)&mom_gamma_even  , 1*sizeof(float) );

    // To use if the correlation parameter is near the boundary of the support
    cudaMalloc( (void**)&Sign_rho    , HK*sizeof(float) );
    cudaMemset( Sign_rho , 1.0f , HK*sizeof(float));
    
    cudaMalloc( (void**)&kin_part  , NB*sizeof(double) );   // Holds the partial sums for the kinetic energy in the device
    double *Hkin_part = (double*) malloc(NB*sizeof(double));
    Kinetic = (double*) calloc(2,sizeof(double));
    Hamiltonian = (double*) calloc(2,sizeof(double));
    
    // Uniform [0,1] numbers for the accept/reject step in Burnin and HMC respectively
    cudaMalloc( (void**)&ub  , Burnin*sizeof(float) );  
    cudaMalloc( (void**)&um  , Iterations*sizeof(float) );
    float *Hub = (float*) malloc(Burnin     *sizeof(float));
    float *Hum = (float*) malloc(Iterations *sizeof(float));

    // Poisson(50) number of leapfrog steps in HMC
    cudaMalloc( (void**)&d_mSteps  , Iterations *sizeof(unsigned int));
    unsigned int *mSteps = (unsigned int*) malloc(Iterations *sizeof(unsigned int));
    
    error = cudaGetLastError();
    printf("\n  Memory allocation error: %s\n", cudaGetErrorString(error));
    cudaDeviceSynchronize();


    curandGenerateUniform(gen , ub , Burnin);
    cudaDeviceSynchronize();
    curandGenerateUniform(gen , um , Iterations);
    cudaDeviceSynchronize();
    cudaMemcpy(Hub,ub,Burnin*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(Hum,um,Iterations*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    curandGeneratePoisson(gen , d_mSteps , Iterations , L);
    cudaDeviceSynchronize();
    cudaMemcpy(mSteps,d_mSteps,Iterations  *sizeof(unsigned int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    
    float rate_c , rate_c_mult;

    error = cudaGetLastError();
    printf("\n  HMC random number generation errors: %s\n", cudaGetErrorString(error));
    printf("\nSTEP 5: The memory has been allocated and the algorithm is about to start.\n");
    fflush(NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    

    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PART 6: HMC 
   
    // Allocate the Mass vectors of the scalar parameters
    cudaMalloc( (void**)&Mass_sigma  , HK*sizeof(float) );
    cudaMalloc( (void**)&Mass_rho    , HK*sizeof(float) );
    cudaMalloc( (void**)&Mass_beta   , HK_star*sizeof(float) );
    float *HMass_sigma   = (float*) calloc(HK,sizeof(float));
    float *HMass_rho     = (float*) calloc(HK,sizeof(float));
    float *HMass_beta    = (float*) calloc(HK_star,sizeof(float));
    // Set the square root masses to some initial good guess
    for (i=0 ; i<HK ; i++) {
    	HMass_beta[i] = tmp_mass[0];
    }
    for (i=HK ; i<HK_star ; i++) {
        HMass_beta[i] = tmp_mass[3];
    }
    for (i=0 ; i<HK ; i++) {
    	HMass_sigma[i] = tmp_mass[1];
    	HMass_rho[i]   = tmp_mass[2];
    }
    HMass_rho[2] = 1.0f;
    HMass_sigma[2] = 4.0f;
    HMass_beta[2] = 10.0f;
    HMass_beta[3] = 20.0f;

    // And move those to the GPU as well
    cudaMemcpy(Mass_sigma,HMass_sigma,HK*sizeof(float)      ,cudaMemcpyHostToDevice);
    cudaMemcpy(Mass_rho,  HMass_rho,  HK*sizeof(float)      ,cudaMemcpyHostToDevice);
    cudaMemcpy(Mass_beta ,HMass_beta, HK_star*sizeof(float) ,cudaMemcpyHostToDevice);

    // Here I allocate memory for some variables that will be used for MCMC purposes
    accepted=0; // For adaptation of the stepsize during Phase I of the algorithm
    int *accept_history = (int*) calloc(Burnin,sizeof(int));
    float *epsilon_history = (float*) calloc(Burnin,sizeof(float));

    float *big =   (float*)  calloc(HK*V_extended,sizeof(float));
    
    // Before the first iteration, some quantities need to be calculated
    FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
    FIND_ZB(Zc,beta,ZB,COVARIATES);
	FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
    FIND_LIK(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol,lik_part,Hlik_part,ZB,HZB,Hsigma,beta,Hbeta,likelihood,Hcounts);
    SAVE_ORIGINALS(likelihood,beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);




    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Phase I
    // The long vectors are updated to shape up the instensity  
    for (i=0 ; i<Burngamma ; i++) {
        curandGenerateNormal(gen, mom_gamma, K*V    , 0.0f, 1.0f);
        cudaDeviceSynchronize();
        FIND_KINETIC_GAMMA(mom_gamma , Kinetic , kin_part , Hkin_part);
        Kinetic[1] = Kinetic[0];
        FIND_ZB(Zc,beta,ZB,COVARIATES);
        Update<<<HK*V/NT,NT>>>(mom_gamma,grad_gamma,0.5f*epsilon);
        cudaDeviceSynchronize();
		for (j=0 ; j<L ; j++) {

            Update<<<HK*V/NT,NT>>>(gamma,mom_gamma,epsilon); // Full step for the parameters
            cudaDeviceSynchronize();
            FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
            FIND_ZB(Zc,beta,ZB,COVARIATES);
            FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
            Update<<<HK*V/NT,NT>>>(mom_gamma,grad_gamma,epsilon); // Full step for the momentum
            cudaDeviceSynchronize();
        }
		Update<<<HK*V/NT,NT>>>(gamma,mom_gamma,epsilon); // Full step for the parameters 
        cudaDeviceSynchronize();
        FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
        FIND_ZB(Zc,beta,ZB,COVARIATES);
        FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
        Update<<<HK*V/NT,NT>>>(mom_gamma,grad_gamma,0.5f*epsilon); // Half a step for the momentum
        cudaDeviceSynchronize();
        // End of leapfrog integration.
 		// Now do the Metropolis step
        FIND_LIK(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol,lik_part,Hlik_part,ZB,HZB,Hsigma,beta,Hbeta,likelihood,Hcounts);
        FIND_KINETIC_GAMMA(mom_gamma , Kinetic , kin_part , Hkin_part);
        Hamiltonian[0] = -likelihood[0]+Kinetic[0];
        Hamiltonian[1] = -likelihood[1]+Kinetic[1];
        ratio = exp(Hamiltonian[1]-Hamiltonian[0]);
        if (Hub[i]<ratio) {
            accepted +=1 ;
            SAVE_ORIGINALS(likelihood,beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        } else { 
            REVERT_STATE(beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        }
		// Now we can adjust the stepsize.
        if ( (i%Adjust)==0 ) {
            if (i>0) {
                rate_c = (float)accepted/(float)Adjust;
                if (rate_c<0.55f)
                    rate_c_mult = 0.9f;
                if (rate_c>0.75f)
                    rate_c_mult = 1.1f;
	            if (rate_c>=0.55f & rate_c<=0.75f)
	               rate_c_mult = 1.0f;
                epsilon *= rate_c_mult;
                accepted = 0;
            }
        }
		// Every 25 iterations print some quantities to see how things are going 
        if ((i%25)==0) {
			printf("\n BURNIN (VECTORS) ITERATION %d | stepsize %.10f | ",i,epsilon);
     	    fflush(NULL);
     	}
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Phase II: the burnin
	accepted = 0;
    epsilon = 0.00001f;
    // Open the txt file that will hold the sums
    FILE *HMCI  = fopen("./outputs/burnin.txt", "w");
    FILE *RFX   = fopen("./outputs/rfx.txt", "w");
 
    // Generate random uniform numbers for the Metropolis-Hastings ratio
    curandGenerateUniform(gen , ub , Burnin);
    cudaMemcpy(Hub,ub,Burnin*sizeof(float),cudaMemcpyDeviceToHost);

    // Set all the accept history equal to zero
    for (i=0 ; i<Burnin ; i++) {
        accept_history[i]=0;
    }

	// Run the HMC burnin
    for (i=0 ; i<Burnin ; i++) {
    	GENERATE_MOMENTUM(mom_beta , mom_sigma , mom_gamma , gen , mom_beta_even , mom_sigma_even , mom_gamma_even , Mass_sigma , Mass_beta, mom_rho, mom_rho_even, Mass_rho);
        FIND_KINETIC(mom_beta , mom_sigma , mom_gamma , Kinetic , kin_part,Hkin_part , Hmom_beta , Hmom_sigma , HMass_sigma , HMass_beta,mom_rho,Hmom_rho,HMass_rho);
        Kinetic[1] = Kinetic[0];
        FIND_ZB(Zc,beta,ZB,COVARIATES);
        // Start the Leapfrog integration
        UPDATE_MOMENTUM(0.5f*epsilon , mom_beta , grad_beta , mom_sigma , grad_sigma , mom_gamma , grad_gamma , mom_rho , grad_rho , Sign_rho);
        for (j=0 ; j<L ; j++) {

            UPDATE_PARAMS( epsilon , beta , mom_beta , sigma , mom_sigma , gamma , mom_gamma , Mass_sigma , Mass_beta , rho,Hrho,mom_rho,Mass_rho,Sign_rho);
            FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
            FIND_ZB(Zc,beta,ZB,COVARIATES);
            FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
            UPDATE_MOMENTUM( epsilon , mom_beta , grad_beta , mom_sigma , grad_sigma , mom_gamma , grad_gamma, mom_rho , grad_rho , Sign_rho);
        
        }
		UPDATE_PARAMS( epsilon , beta , mom_beta , sigma , mom_sigma , gamma , mom_gamma , Mass_sigma , Mass_beta, rho,Hrho,mom_rho,Mass_rho,Sign_rho);
        FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
        FIND_ZB(Zc,beta,ZB,COVARIATES);
        FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
        UPDATE_MOMENTUM(0.5f*epsilon , mom_beta , grad_beta , mom_sigma , grad_sigma , mom_gamma , grad_gamma, mom_rho , grad_rho , Sign_rho);
        // Leapfrog has just finished
        // Metropolis step
        FIND_LIK(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol,lik_part,Hlik_part,ZB,HZB,Hsigma,beta,Hbeta,likelihood,Hcounts);
        FIND_KINETIC(mom_beta , mom_sigma , mom_gamma , Kinetic , kin_part,Hkin_part , Hmom_beta , Hmom_sigma , HMass_sigma , HMass_beta,mom_rho,Hmom_rho,HMass_rho);
        Hamiltonian[0] = -likelihood[0]+Kinetic[0];
        Hamiltonian[1] = -likelihood[1]+Kinetic[1];
        ratio = exp(Hamiltonian[1]-Hamiltonian[0]);
        if (Hub[i]<ratio) {
            accept_history[i] = 1;
            SAVE_ORIGINALS(likelihood,beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        } else { 
            REVERT_STATE(beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        }
        /* Update the random effects */
        if ((i%5)==0) {
            FIND_ZB(Zc,beta,ZB,COVARIATES);
            UPDATE_RFX(Hrfx ,rfx, HZB, ZB, Hcounts, sigma, Cgamma, vol, COVARIATES, RNG, author);
            FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
            FIND_LIK(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol,lik_part,Hlik_part,ZB,HZB,Hsigma,beta,Hbeta,likelihood,Hcounts);
            SAVE_ORIGINALS(likelihood,beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        }
        // Adjust the stepsize to 75%
        if ( ((i%Adjust)==0) && (i>Adjust_window) ) {
            accepted = 0;
            for (j=0 ; j<Adjust_window ; j++){
                accepted += accept_history[i-Adjust_window+1+j];
            }
            rate_c = (float)accepted/(float)Adjust_window;
            if (rate_c<0.60f) {
                rate_c_mult = 0.95f;
	        }
            if (rate_c>0.80f){
                rate_c_mult = 1.05f;
            }
	        if (rate_c>=0.55f & rate_c<=0.75f){
	           	rate_c_mult = 1.0f;
            }
            epsilon *= rate_c_mult;
        }
        
		// Every one iteration save the scalars
        if ((i%1)==0) {
            // Bring everything to the host
            cudaMemcpy(Hsigma,ori_sigma,HK*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(Hrho  ,ori_rho  ,HK*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(Hbeta,ori_beta,HK_star*sizeof(float),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            // And now print scalar parameters to the file
            fprintf(HMCI,"%.8f ",epsilon);
            for (kk=0 ; kk<HK ; kk++){
                fprintf(HMCI,"%.8f ",Hsigma[kk]);
            }
            for (kk=0 ; kk<HK ; kk++){
                fprintf(HMCI,"%.8f ",Hrho[kk]);
            }
            for (kk=0 ; kk<HK_star ; kk++){
                fprintf(HMCI,"%.8f ",Hbeta[kk]);
            }
            fprintf(HMCI,"%.8lf %.8lf ",likelihood[1],Hamiltonian[1]);
            fprintf(HMCI,"%d\n",i);
            fflush(NULL);
        }
        /* SAVE RANDOM EFFECTS */
        for (kk=0 ; kk<HI ; kk++){
            fprintf(RFX,"%.8f ",Hrfx[kk]);
        }
        fprintf(RFX,"%d\n",i);
    	// Print something every  now and then
        if ((i%50)==0) {
			printf("\n BURNIN ITERATION %d | stepsize %.10f | ",i,epsilon);
            fflush(NULL);
     	}

        // Save the value of the stepsize
        epsilon_history[i] = epsilon;
        
        /* Every 500 iterations make a snapshot of the parameters */
        if ((i%500)==0) {
            SNAPSHOT(Hsigma,ori_sigma,Hrho,ori_rho,Hbeta,ori_beta,big,ori_gamma);
        }
    }

    // Close the txt file that holds the traceplots of burnin
    fclose(RFX);
    fclose(HMCI);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Phase III: HMC
    // Create the file that will hold the scalars
    FILE *HMCII  = fopen("./outputs/hmc.txt", "w");
    FILE *RFXII  = fopen("./outputs/alpha.txt", "w");

    // Allocate depending on whether type comparisons are made
    int ngp;
    if (Diff==0) {
        ngp = HK; // i.e. as many as the spatially varying coefficients
    } else {
        ngp = HK + HK*(HK-1)/2; // i.e. as many as the spatially varying coefficients and choose(HK,2)
    }
    double *Hgp_s = (double*) calloc(ngp*V_extended,sizeof(double));  /* for the running sums */
    cudaMalloc((void**)&gp_s ,sizeof(double)*V_extended*ngp); 
    cudaMemcpy(gp_s,Hgp_s,sizeof(double)*ngp*V_extended,cudaMemcpyHostToDevice);
    double *Hgp_ss = (double*) calloc(ngp*V_extended,sizeof(double)); /* for the running sums of squares */
    cudaMalloc((void**)&gp_ss,sizeof(double)*V_extended*ngp);  
    cudaMemcpy(gp_ss,Hgp_ss,sizeof(double)*ngp*V_extended,cudaMemcpyHostToDevice);
    int nsamples=0; 
    double gp_sum[2];

    

    /* For the HMC use the average stepsize of the burn-in period */
    rate_c_mult = 0.0f;
    for (i=500 ; i<Burnin ; i++) {
        rate_c_mult += epsilon_history[i];
        nsamples++;
    }
    epsilon = rate_c_mult/(float)nsamples;
    nsamples=0;


    for (i=0 ; i<Iterations ; i++) {
		GENERATE_MOMENTUM(mom_beta , mom_sigma , mom_gamma , gen , mom_beta_even , mom_sigma_even , mom_gamma_even , Mass_sigma , Mass_beta, mom_rho, mom_rho_even, Mass_rho);
        FIND_KINETIC(mom_beta , mom_sigma , mom_gamma , Kinetic , kin_part,Hkin_part , Hmom_beta , Hmom_sigma , HMass_sigma , HMass_beta,mom_rho,Hmom_rho,HMass_rho);
        Kinetic[1] = Kinetic[0];
        FIND_ZB(Zc,beta,ZB,COVARIATES);
        // Start the Leapfrog integration
        UPDATE_MOMENTUM(0.5f*epsilon , mom_beta , grad_beta , mom_sigma , grad_sigma , mom_gamma , grad_gamma, mom_rho , grad_rho , Sign_rho);
        for (j=0 ; j<mSteps[i] ; j++) {

            UPDATE_PARAMS( epsilon , beta , mom_beta , sigma , mom_sigma , gamma , mom_gamma , Mass_sigma , Mass_beta, rho,Hrho,mom_rho,Mass_rho,Sign_rho);
            FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
            FIND_ZB(Zc,beta,ZB,COVARIATES);
            FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
            UPDATE_MOMENTUM(epsilon , mom_beta , grad_beta , mom_sigma , grad_sigma , mom_gamma , grad_gamma, mom_rho , grad_rho , Sign_rho);
        }
		UPDATE_PARAMS( epsilon , beta , mom_beta , sigma , mom_sigma , gamma , mom_gamma , Mass_sigma , Mass_beta, rho,Hrho,mom_rho,Mass_rho,Sign_rho);
        FIND_EIGEN_PROD (rho,DFT_big,dist,plan_big,eigen,gamma,DFT,plan,Cgamma,CSgamma);
        FIND_ZB(Zc,beta,ZB,COVARIATES);
        FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
        UPDATE_MOMENTUM(0.5f*epsilon , mom_beta , grad_beta , mom_sigma , grad_sigma , mom_gamma , grad_gamma, mom_rho , grad_rho , Sign_rho);
        // Leapfrog has just finished
        // Metropolis step
        FIND_LIK(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol,lik_part,Hlik_part,ZB,HZB,Hsigma,beta,Hbeta,likelihood,Hcounts);
        FIND_KINETIC(mom_beta , mom_sigma , mom_gamma , Kinetic , kin_part,Hkin_part , Hmom_beta , Hmom_sigma , HMass_sigma , HMass_beta,mom_rho,Hmom_rho,HMass_rho);
        Hamiltonian[0] = -likelihood[0]+Kinetic[0];
        Hamiltonian[1] = -likelihood[1]+Kinetic[1];
        ratio = exp(Hamiltonian[1]-Hamiltonian[0]);
        if (Hum[i]<ratio) {
            SAVE_ORIGINALS(likelihood,beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        } else { 
            REVERT_STATE(beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        }
        /* Update the random effects */
        if ((i%5)==0) {
            FIND_ZB(Zc,beta,ZB,COVARIATES);
            UPDATE_RFX(Hrfx ,rfx, HZB, ZB, Hcounts, sigma, Cgamma, vol, COVARIATES, RNG, author);
            FIND_GRAD(gamma,grad_gamma,beta,grad_beta,sigma,grad_sigma,voxel_tmp,ZB,Cgamma,vol,sigma_part,beta_part,Hsigma_part,Hbeta_part,COVARIATES,TERM_GAMMA,DFT,eigen,plan,Hsigma,Hbeta,HZN,Hrho,rho,CSgamma,rho_part,Hrho_part,grad_rho);
            FIND_LIK(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol,lik_part,Hlik_part,ZB,HZB,Hsigma,beta,Hbeta,likelihood,Hcounts);
            SAVE_ORIGINALS(likelihood,beta,ori_beta,sigma,ori_sigma,gamma,ori_gamma,Cgamma,ori_Cgamma,grad_beta,ori_grad_beta,grad_sigma,ori_grad_sigma,grad_gamma,ori_grad_gamma,rho,ori_rho,grad_rho,ori_grad_rho,CSgamma,ori_CSgamma);
        }
		// Save the scalars every now and then 
        if ((i%1)==0) {
            // Bring everything to the host
            cudaMemcpy(Hsigma,ori_sigma,HK*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(Hrho  ,ori_rho  ,HK*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(Hbeta,ori_beta,HK_star*sizeof(float),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            // And now print to the file
            fprintf(HMCII,"%.8f ",epsilon);
            for (kk=0 ; kk<HK ; kk++){
                fprintf(HMCII,"%.8f ",Hsigma[kk]);
            }
            for (kk=0 ; kk<HK ; kk++){
                fprintf(HMCII,"%.8f ",Hrho[kk]);
            }
            for (kk=0 ; kk<HK_star ; kk++){
                fprintf(HMCII,"%.8f ",Hbeta[kk]);
            }
            fprintf(HMCII,"%.8lf %.8lf ",likelihood[1],Hamiltonian[1]);
            fprintf(HMCII,"%d\n",i);
            fflush(NULL);
        }

        /* Write random effects to the file */
        for (kk=0 ; kk<HI ; kk++){
            fprintf(RFXII,"%.8f ",Hrfx[kk]);
        }
        fprintf(RFXII,"%d\n",i);

    	// Print something every 25 iterations to monitor the progress
        if ((i%50)==0) {
			printf("\n HMC ITERATION %d ",i); fflush(NULL);
     	}

     	// Save samples for the GPs
     	if ((i%Save)==0) {
            snprintf( filename, sizeof(char) * 32, "./outputs/gps/gp_%i.txt", i);
            FIELDS  = fopen(filename , "w");
            cudaMemcpy(big  , ori_Cgamma , HK*V_extended*sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
     		for (ii=0 ; ii<V_extended ; ii++) {
                if (brain_extended[ii]==1) {
                    for (kk=0 ; kk<HK ; kk++) {
                        fprintf(FIELDS,"%.7f ",big[kk*V_extended+ii]);
                    }
                    fprintf(FIELDS,"\n");
                }    
            }
            fclose(FIELDS);
     	}

        // Save the running sums of the GPs
        if ((i%Thinning)==0) {
            if (Diff==0) {
                saveGPs<<<V/NT,NT>>>(gp_s,gp_ss,ori_Cgamma,ori_beta,ori_sigma);
                cudaDeviceSynchronize();
                nsamples++;
            } else {
                saveGPsDiff<<<V/NT,NT>>>(gp_s,gp_ss,ori_Cgamma,ori_beta,ori_sigma);
                cudaDeviceSynchronize();
                nsamples++;
            }
        }

        // Give some output every 500 iterations
        if ((i%500)==0) {
            /* Snapshot of the model parameters */
            SNAPSHOT(Hsigma,ori_sigma,Hrho,ori_rho,Hbeta,ori_beta,big,ori_gamma);
            
            /* GP summaries */
            cudaMemcpy(Hgp_s, gp_s, ngp*V_extended*sizeof(double),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaMemcpy(Hgp_ss,gp_ss,ngp*V_extended*sizeof(double),cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            /* voxel-wise mean and variance */
            FIELDS  = fopen("./outputs/gp_summaries.txt", "w");
            for (ii=0 ; ii<V_extended ; ii++) {
                if (brain_extended[ii]==1) {
                    for (kk=0 ; kk<HK ; kk++){
                        // using Likelihood to store find the mean and variance 
                        gp_sum[0] = Hgp_s[kk*V_extended+ii]/nsamples;
                        gp_sum[1] = Hgp_ss[kk*V_extended+ii] - (pow(Hgp_s[kk*V_extended+ii],2.0)/(double)nsamples) ;
                        gp_sum[1] = gp_sum[1]/((double)nsamples-1);
                        fprintf(FIELDS,"%.7f %.7f ",gp_sum[0],gp_sum[1]);
                    }
                    fprintf(FIELDS,"\n"); fflush(NULL);
                }    
            }
            fclose(FIELDS);
            /* voxel-wise mean standardised posterior difference */
            if (Diff==1) {
                FIELDS  = fopen("./outputs/gp_diff.txt", "w");
                for (ii=0 ; ii<V_extended ; ii++) {
                    if (brain_extended[ii]==1) {
                        for (kk=HK ; kk<ngp ; kk++){
                            // find the mean standardised posterior difference
                            gp_sum[0] = Hgp_s[kk*V_extended+ii]/nsamples;
                            gp_sum[1] = Hgp_ss[kk*V_extended+ii] - (pow(Hgp_s[kk*V_extended+ii],2.0)/(double)nsamples);
                            gp_sum[1] = sqrt( gp_sum[1]/((double)nsamples-1) );
                            gp_sum[0] /= gp_sum[1];
                            fprintf(FIELDS,"%.7f ",gp_sum[0]);
                            fprintf(FIELDS,"\n"); fflush(NULL);
                        }
                    }    
                }
                fclose(FIELDS);
            } 
        }
    }

    // Close the file of chains
    fclose(RFXII);
    fclose(HMCII);
    printf("\nSTEP 6: HMC completed.\n");
    fflush(NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    


    
    /* Free the reserved memory */ 
    cudaFree(gp_ss); free(Hgp_ss); cudaFree(gp_s); free(Hgp_s); free(big);
    free(epsilon_history); free(accept_history); free(HMass_beta); free(HMass_rho); 
    free(HMass_sigma); cudaFree(Mass_beta); cudaFree(Mass_rho); cudaFree(Mass_sigma);

    free(mSteps); cudaFree(d_mSteps);  free(Hum); free(Hub); cudaFree(um); cudaFree(ub);
    free(Hamiltonian); free(Kinetic); free(Hkin_part); cudaFree(kin_part);
    cudaFree(Sign_rho); cudaFree(mom_gamma_even); cudaFree(mom_rho_even); cudaFree(mom_sigma_even); cudaFree(mom_beta_even);  free(Hmom_rho); free(Hmom_sigma); free(Hmom_beta); 
    cudaFree(mom_gamma); cudaFree(mom_rho); cudaFree(mom_sigma); cudaFree(mom_beta); 
    cudaFree(ori_grad_gamma); cudaFree(ori_grad_rho); cudaFree(ori_grad_sigma); cudaFree(ori_grad_beta);
    cudaFree(ori_CSgamma); cudaFree(ori_Cgamma); cudaFree(ori_gamma); cudaFree(ori_rho); cudaFree(ori_sigma);  cudaFree(ori_beta);
    curandDestroyGenerator(gen);
    free(likelihood); free(Hlik_part); cudaFree(lik_part); cudaFree(lik_tmp);
    free(Hbeta_part); free(Hrho_part); free(Hsigma_part);  cudaFree(beta_part); cudaFree(rho_part); cudaFree(sigma_part); 
    cudaFree(TERM_GAMMA); free(HTERM_GAMMA); cudaFree(voxel_tmp);
    cudaFree(grad_beta); cudaFree(grad_rho); cudaFree(grad_sigma);  cudaFree(grad_gamma);
    cudaFree(vol); free(Hvol); 

    cublasShutdown();
    cudaFree(ZB); free(HZB); free(HZN);
    /* RFX CHANGE */
    free(Hrfx); cudaFree(rfx);
    /* */
    free(Hgamma); cudaFree(gamma); free(Hbeta); cudaFree(beta); free(Hsigma); cudaFree(sigma); free(Hrho); cudaFree(rho);
    cudaFree(CSgamma); cudaFree(Cgamma); cudaFree(eigen);
    cufftDestroy(plan); cufftDestroy(plan_big);
    cudaFree(DFT); cudaFree(DFT_big);
    cudaFree(dist); free(Hdist); free(HID); free(brain_extended); free(grid_extended); free(Hgrid);
    cudaFree(COVARIATES); free(HCOVARIATES); cudaFree(Zc); free(HZc); free(brain); free(author); free(Hcounts);
    for (i=0 ; i<HI ; i++){
        free(HZ[i]);
    }
    free(HZ);
    for (i=0 ; i<HN ; i++){
        free(Hfoci[i]);
    }
    free(Hfoci);
    free(RNG);
    
    printf("\n END OF SIMULATION \n");
    exit(0);
}


