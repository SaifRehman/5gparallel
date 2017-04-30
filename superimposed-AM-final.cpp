#include "mconvert.h"
#include <iostream>
#include <armadillo>
using namespace arma ;
void initialise();
void mult_vec_by_scalar(float scale_val,cx_mat *in_vec,cx_mat *out_vec);
void fill_cmplx_vector(float scale_val, int scale_var, int constant, cx_mat *in_mat, cx_mat *out_vec);
void mult_vec_by_vec(cx_mat *in_mat, cx_mat *in_vec1, cx_mat *out_vec);
void initilialise_mean_matrixes_with_zeros(int scale_var, int Channels, cx_mat *out_mat);
void append_mat_with_zeros(int scalar_var, int constant, cx_mat *in_vec, cx_mat *out_vec);
void rotate_matrix(int, cx_mat *in_vec, cx_mat *out_mat);
void calculate_scalar_AandAd(float constant, float var, int scale_var1, int scale_var2, int scale_var3, int variable, int &out);
void calculate_cmplx_matrix_C(int N_size, int scalar, cx_mat *in_mat, cx_mat *out_mat);
void compute_matrix_inverse(cx_mat *in_mat, cx_mat *out_mat);
void calculate_cmplx_matrix_Q(int scale_var, int var, int N_size, int scalar, cx_mat *in_mat1, cx_mat *in_mat2, cx_mat *out_mat);
void calculate_cmplx_matrix_wandw1(int variable, int N_size, int Runs, cx_mat *out_mat);
void calculate_cmplx_matrix_mu(int scalar, cx_mat *in_mat1 ,cx_mat *in_mat2, cx_mat *in_vec1, cx_mat *in_mat3, cx_mat *in_vec2, cx_mat *in_vec3 , cx_mat *out_mat);
void fill_mean_matrices(int Runs, cx_mat *out_mat);
void initialise_variable_values();
void obtain_SB_BFGS_real_efficient_plus(cx_mat initial_estimate, cx_mat v, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat& BFGS_estimate, int& SBcount, cx_mat& LScount);
vec obtain_gradient_real_efficientv8(cx_mat estimate, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat F, cx_mat y, cx_mat zk_matrix);
void ML_linesearch_BFGS_real_efficientv2(cx_mat estimate, vec gradient, vec delta, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat F, cx_mat y, cx_mat zk_matrix, int& t, int& count);
float ML_eval_SB_non_new_efficient(cx_mat estimate, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat F, cx_mat y, cx_mat zk_matrix);
mat generate_IDFT(int L);

int main(int argc, char** argv)
{
  cx_mat B1, B2, B3, C, Cinv, F, G1, H1, H2, H2_g1, H2_h1, LScount, La, Lambda1, Lambda2, Lambda3, Lb, Q, Qinv, X1, X2, X3, g1, g1_app, g1_h, h1, h1_app, h1_h, h2, h2_app, h2_g1, h2_h, h2_h1, initial_estimate, mu, n, n1, q, q_h, s1, s1_matrix, s2, s2_matrix, t1, t2, t3, v, w, w1 , y, zk ;
  cx_mat w_matrix = randu<cx_mat>(1,16);
  cx_mat w1_matrix = randu<cx_mat>(1,16);

  w_matrix(1,1) = cx_double(-0.0601, 0.0140);
  w_matrix(1,2) = cx_double(0.0346, 0.0230);
  w_matrix(1,3) = cx_double(-0.0207 , 0.0045);
  w_matrix(1,4) = cx_double(-0.0161 , 0.0193);
  w_matrix(1,5) = cx_double(0.0186 , 0.0348);
  w_matrix(1,6) = cx_double(-0.0127 , 0.0029);
  w_matrix(1,7) = cx_double(-0.0223 , 0.0183);
  w_matrix(1,8) = cx_double(0.0143 , 0.0065);
  w_matrix(1,9) = cx_double(-0.0057 , 0.0253);
  w_matrix(1,10) = cx_double(-0.0251 , 0.0197);
  w_matrix(1,11) = cx_double(0.0368 , 0.0211);
  w_matrix(1,12) = cx_double(-0.0069 , 0.0139);
  w_matrix(1,13) = cx_double(-0.0173 , 0.0445);
  w_matrix(1,14) = cx_double(-0.0303 , 0.0213);
  w_matrix(1,15) = cx_double(-0.0073 , 0.0058);
  w_matrix(1,16) = cx_double(0.0347 , 0.0268);

  w1_matrix(1,1) = cx_double(0.0248 , 0.0243);
  w1_matrix(1,2) = cx_double(-0.0361 , 0.0380);
  w1_matrix(1,3) = cx_double(-0.0080 , 0.0289);
  w1_matrix(1,4) = cx_double(-0.0002 , 0.0122);
  w1_matrix(1,5) = cx_double(-0.0003 , 0.0477);
  w1_matrix(1,6) = cx_double(-0.0127 , 0.0029);
  w1_matrix(1,7) = cx_double(0.0040 , 0.0185);
  w1_matrix(1,8) = cx_double(-0.0116 , 0.0361);
  w1_matrix(1,9) = cx_double(-0.0054 , 0.0272);
  w1_matrix(1,10) = cx_double(-0.0155 , 0.0361);
  w1_matrix(1,11) = cx_double(-0.0153 , 0.0134);
  w1_matrix(1,12) = cx_double(0.0209 , 0.0233);
  w1_matrix(1,13) = cx_double(-0.0270 , 0.0055);
  w1_matrix(1,14) = cx_double(-0.0081 , 0.0332);
  w1_matrix(1,15) = cx_double(-0.0264 , 0.0080);
  w1_matrix(1,16) = cx_double(0.0144 , 0.0079);

  double current_MSE ;
  float A, Ad, E, Pr, Pt, SNR, a, a_h, alfa, b, b_h, d, dp, var ;
  int K, L, Lg1, Lh1, Lh2, Lq, M, N, Num_channels, Num_runs, SNR_index, ch_index, k, k1, k2, run ;
  mat Average_Linesearch, Average_SDSB, LSNR, MSE_LS_matrix, MSE_LS_runs, MSE_LS_vector, MSE_SDSB_matrix, MSE_SDSB_vector ;
  rowvec Linesearch_runs, MSE_SDSB_runs ;
  vec S, SBcount, SDSB_estimate, SDSBcount_runs, SNR_vector, var_vector ;
  // all values of snr_vector
  Lg1 = 2 ;// should change
  Lh1 = 2 ;//should change
  Lh2 = 2 ;//should change
  M = 4 ;
  E = 1 ;
  Pr = 2*E ;
  d = sqrt(6*E/(M-1)) ;
  dp = d/2.0 ;
  N = 16 ; //should change
  K = 1 ;
  L = 10 ;// to be changed
  k1 = 20 ;// to be changed
  k2 = 40 ;// to be changed
  Num_channels = 1 ;
  alfa = 0.55 ;
  Num_runs = 10 ;
  SNR_vector = m2cpp::fspan(0, 3, 30) ;// should change, [1:2:10], [20] vec comp
  LSNR = m2cpp::length(SNR_vector) ;
  var_vector = (E*arma::pow(10, (-SNR_vector/10.0))) ;
  //load("OFDM_nondecaying_channels.mat") ;
  mult_vec_by_scalar(E,&X1,N);
  S = arma::trans((m2cpp::span<rowvec>(1, N))) ;
  fill_cmplx_vector(E,N, k1, &S, &X2);
  Pt = Pr*alfa ;
  fill_cmplx_vector(Pt,N, k2, &S, &X3);
  F = generate_IDFT(N);
  mult_vec_by_vec(&F, &X1, &t1);
  mult_vec_by_vec(&F, &X2, &t2);
  mult_vec_by_vec(&F, &X3, &t3);
  initilialise_mean_matrixes_with_zeros(LSNR, Num_channels, &MSE_SDSB_matrix);
  initilialise_mean_matrixes_with_zeros(LSNR, Num_channels, &MSE_LS_matrix);
  initilialise_mean_matrixes_with_zeros(LSNR, Num_channels, &Average_SDSB);
  initilialise_mean_matrixes_with_zeros(LSNR, Num_channels, &Average_Linesearch);
  for (ch_index=1; ch_index<=Num_channels; ch_index++)
  {
    h1 = h1_matrix.cols(ch_index) ;
    h2 = h2_matrix.cols(ch_index) ;
    g1 = g1_matrix.cols(ch_index) ;
    append_mat_with_zeros(N, Lh1, &h1, &h1_app);
    append_mat_with_zeros(N, Lh2, &h2, &h2_app);
    append_mat_with_zeros(N, Lg1, &g1, &g1_app);

    rotate_matrix(size(h1),&h1_app, &H1);
    rotate_matrix(size(h1),&h2_app, &H2);
    rotate_matrix(size(h1),&G1_app, &G1);

    for (SNR_index=1; SNR_index<=LSNR; SNR_index++)
    {
      SNR = SNR_vector(SNR_index) ;
      var = var_vector(SNR_index) ;

      calculate_scalar_AandAd(constant,var, scale_var1,scale_var2, scale_var3, variable, A);
      calculate_scalar_AandAd(0,var, scale_var1,scale_var2, scale_var3, variable, Ad);

      calculate_cmplx_matrix_C(N, A, &H2, &C);
      compute_matrix_inverse(&C, &Cinv);

      calculate_cmplx_matrix_Q(E, var, N, Ad, &H2, &G1, &Q);
      compute_matrix_inverse(&Q, &Qinv);

      calculate_cmplx_matrix_wandw1(var, N, Num_runs, &w_matrix);// get w_matrix from file (load)
      calculate_cmplx_matrix_wandw1(var, N, Num_runs, &w1_matrix);// even this //

      calculate_cmplx_matrix_mu(A, &H2, &H1, &t1, &G1, &t2, &t3,&mu);


      rotate_matrix(size(t1),&t1, &B1);
      rotate_matrix(size(t2),&t2, &B2);
      rotate_matrix(size(t3),&t3, &B3);

      Lambda1 = B1.cols(m2cpp::span<uvec>(1, Lh1+Lh2-1)) ;
      Lambda2 = B2.cols(m2cpp::span<uvec>(1, Lg1+Lh2-1)) ;
      Lambda3 = B3.cols(m2cpp::span<uvec>(1, Lh2)) ;

      fill_mean_matrices(Num_runs,MSE_SDSB_runs);
      fill_mean_matrices(Num_runs,MSE_LS_runs);
      fill_mean_matrices(Num_runs,SDSBcount_runs);
      fill_mean_matrices(Num_runs,Linesearch_runs);

      //STOP Here
      //
      s1_matrix = sqrt(E/2.0)*F*(randn(N, L)+cx_double(0, 1)*randn(N, L)) ; // comment
       // load s1_matrix data

      a = conv(h1, h2) ;
      b = conv(g1, h2) ;
      q = {a, b, h2} ;
      Lq = m2cpp::length(q) ;
      La = Lh1+Lh2-1 ;
      Lb = Lg1+Lh2-1 ;
      for (run=1; run<=Num_runs; run++)
      {
        w = w_matrix.cols(run) ;
        w1 = w1_matrix.cols(run) ;
        y = A*H2*H1*t1+A*H2*G1*t2+H2*t3+A*H2*w+w1 ;
        S = {arma::join_rows(arma::join_rows(A*Lambda1, A*Lambda2), Lambda3)} ;
        q_h = pinv(S)*y ;
        calculate_h_matrix(1,La,&q_h,&a_h);//func
        calculate_h_matrix(La+1,La+Lb,&q_h,&b_h);//func
        calculate_h_matrix(La+Lb+1,Lq,&q_h,&h2_h);//func
        calculate_hs_matrix( Lh1-1,h2_h,h2_h1);//func
        rotate_matrix(size(h2_h1),&h2_h1, &H2_h1);
        h1_h = pinv(H2_h1.cols(m2cpp::span<uvec>(1, Lh1)))*a_h ;
        calculate_hs_matrix( Lg1-1,h2_h, h2_g1);//func
        rotate_matrix(size(h2_g1),&H2_g1, &h2_g1);
        g1_h = pinv(H2_g1.cols(m2cpp::span<uvec>(1, Lg1)))*b_h ;
        initial_estimate = {h1_h, h2_h, g1_h} ;
        MSE_LS_runs(run) = pow(norm(initial_estimate-{h1, h2, g1}), 2) ;
        calculate_s_matrix(E,L,N,&F,&s2_matrix);//func
        v = arma::zeros<vec>((L+1)*N) ;
        v(m2cpp::span<uvec>(1, N)) = y ;
        for (k=1; k<=L; k++)
        {
          s1 = s1_matrix.cols(k) ;
          s2 = s2_matrix.cols(k) ;
          n = sqrt(var/2.0)*(randn(N, 1)+cx_double(0, 1)*randn(N, 1)) ; // load from file
          n1 = sqrt(var/2.0)*(randn(N, 1)+cx_double(0, 1)*randn(N, 1)) ; // load from file
          zk = Ad*H2*H1*s1+Ad*H2*G1*s2+Ad*H2*n+n1 ;
          v(m2cpp::span<uvec>(N*k+1, N*(k+1))) = zk ;
        }
        obtain_SB_BFGS_real_efficient_plus(initial_estimate, v, t1, t2, t3, A, Ad, Lh1, Lh2, Lg1, var, N, L, s1_matrix, E) ;
        SDSBcount_runs(run) = SBcount ;
        Linesearch_runs(run) = LScount ;
        MSE_SDSB_runs(run) = pow(norm(SDSB_estimate-{h1, h2, g1}), 2) ;
        current_MSE = MSE_SDSB_runs(run) ;
      }
      MSE_LS_matrix(SNR_index, ch_index) = mean(MSE_LS_runs) ;
      MSE_SDSB_matrix(SNR_index, ch_index) = mean(MSE_SDSB_runs) ;
      MSE_SDSB_matrix(SNR_index, ch_index) ;
      Average_SDSB(SNR_index, ch_index) = mean(SDSBcount_runs) ;
      Average_Linesearch(SNR_index, ch_index) = mean(Linesearch_runs) ;
    }
  }
  std::cout << "Ellapsed time = " << _timer.toc() << std::endl ;
  MSE_SDSB_vector = mean(MSE_SDSB_matrix, 2) ;
  MSE_LS_vector = mean(MSE_LS_matrix, 2) ;
  return 0 ;
}

void initialise_variable_values()
{
  cx_mat B1, B2, B3, C, Cinv, F, G1, H1, H2, H2_g1, H2_h1, LScount, La, Lambda1, Lambda2, Lambda3, Lb, Q, Qinv, X1, X2, X3, g1, g1_app, g1_h, h1, h1_app, h1_h, h2, h2_app, h2_g1, h2_h, h2_h1, initial_estimate, mu, n, n1, q, q_h, s1, s1_matrix, s2, s2_matrix, t1, t2, t3, v, w, w1, y, zk ;
  double current_MSE ;
  float A, Ad, E, Pr, Pt, SNR, a, a_h, alfa, b, b_h, d, dp, var ;
  int K, L, Lg1, Lh1, Lh2, Lq, M, N, Num_channels, Num_runs, SNR_index, ch_index, k, k1, k2, run ;
  mat Average_Linesearch, Average_SDSB, LSNR, MSE_LS_matrix, MSE_LS_runs, MSE_LS_vector, MSE_SDSB_matrix, MSE_SDSB_vector ;
  rowvec Linesearch_runs, MSE_SDSB_runs ;
  vec S, SBcount, SDSB_estimate, SDSBcount_runs, SNR_vector, var_vector ;
  Lg1 = 4 ;
  Lh1 = 5 ;
  Lh2 = 5 ;
  M = 4 ;
  E = 1 ;
  Pr = 2*E ;
  d = sqrt(6*E/(M-1)) ;
  dp = d/2.0 ;
  N = 64 ;
  K = 1 ;
  L = 10 ;
  k1 = 20 ;
  k2 = 40 ;
  Num_channels = 1 ;
  alfa = 0.55 ;
  Num_runs = 10 ;
}

void mult_vec_by_scalar(int scale_val, cx_mat *out_vec, int num_runs)
{
   out_vec = sqrt(scale_val)*arma::ones<vec>(num_runs) ;
} // generates t1; Nx1 vector, by multiplying scalar F by X1[] Nx1 vector.

void fill_cmplx_vector(float scale_val, int scale_var, int constant, cx_mat *in_mat, cx_mat *out_vec)
{
  out_vec = sqrt(scale_val)*arma::exp(-2*datum::pi*cx_double(0, 1)*(in_mat-1)*constant/scale_var) ;
}

void mult_vec_by_vec(cx_mat in_mat, cx_mat *in_vec1, cx_mat *out_vec)
{
    out_vec = in_mat*in_vec2;
}

void initilialise_mean_matrixes_with_zeros(int scale_var, int Channels, cx_mat *out_mat);
{
   out_mat = arma::zeros<mat>(scale_var, Channels) ;
}

void append_mat_with_zeros(int scalar_var, int constant, cx_mat *in_vec, cx_mat *out_vec)
{
    out_vec = {in_vec, arma::zeros<umat>(constant-scalar_var, 1)} ;
}


void calculate_scalar_AandAd(float constant, float var, int scale_var1, int scale_var2, int scale_var3, int variable, int &out);
{
    out = sqrt((1-constant)*variable/((scale_var1+scale_var2)*scale_var3+variable)) ;
}

void calculate_cmplx_matrix_C(int N_size, int scalar, cx_mat *in_mat, cx_mat *out_mat);
{
    out_mat = var*(pow(scalar, 2)*in_mat1*(arma::trans(in_mat1))+eye(N_size)) ;
}

void compute_matrix_inverse(cx_mat *in_mat, cx_mat *out_mat);
{
    out_mat = pow(in_mat,(-1));
}

void calculate_cmplx_matrix_Q(int scale_var, int var,int N_size,int scalar, cx_mat *in_mat1, cx_mat *in_mat2, cx_mat *out_mat);
{
    out_mat = pow(scalar, 2)*scale_var*in_mat1*in_mat2*arma::trans(in_mat2)*pow(in_mat1, 2)*var*in_mat1*in_mat1*eye(N_size) ;

}

void calculate_cmplx_matrix_wandw1(int variable, int N_size, int Runs, cx_mat *out_mat);
{
    out_mat = sqrt(variable/2.0)*(randn(N_size, Runs)+cx_double(0, 1)*randn(N_size, Runs)) ;
}

void calculate_cmplx_matrix_mu(int scalar, cx_mat *in_mat1 ,cx_mat *in_mat2, cx_mat *in_vec1, cx_mat *in_mat3, cx_mat *in_vec2, cx_mat *in_vec3 , cx_mat *out_mat);
{
   out_mat = in_mat1*in_mat2*in_vec1*in_mat3+in_mat1*in_mat2*in_vec2*in_vec3+in_mat1*in_vec7;
}

void fill_mean_matrices(int Runs, cx_mat *out_mat)
{
    out_mat = arma::zeros<rowvec>(Runs) ;
}
void obtain_SB_BFGS_real_efficient_plus(cx_mat initial_estimate, cx_mat v, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat& BFGS_estimate, int& SBcount, cx_mat& LScount)
{
  cx_mat F, current_estimate, y, zk_matrix ;
  float rhok, t ;
  int LT ;
  mat Hk ;
  rowvec LScountVec ;
  vec current_gradient, pk, previous_gradient, sk, yk ;
  y = v(arma::span(0, N-1)) ;
  zk_matrix = reshape(v(arma::span(N, (L+1)*N-1)), N, L) ;
  F = generate_IDFT(N) ; // as doctor saeed
  current_estimate = initial_estimate ;
  SBcount = 0 ;
  current_gradient = obtain_gradient_real_efficientv8(current_estimate, t1, t2, t3, A, Ad, Lh1, Lh2, Lg1, var, N, L, s1_matrix, E, F, y, zk_matrix) ;
  LT = Lh1+Lh2+Lg1 ;
  Hk = 0.005*arma::eye<mat>(2*LT, 2*LT) ;
  LScountVec = arma::zeros<rowvec>(100) ;
  while (pow(norm(current_gradient), 2)>1*pow(10, (-4)))
  {
    SBcount = SBcount+1 ;
    if (SBcount==100)
    {
      SBcount ;
      BFGS_estimate = current_estimate ;
      LScountVec = LScountVec(arma::span(0, SBcount-2)) ;
      LScount = mean(LScountVec) ;
      return ;
    }
    pk = -Hk*current_gradient ;
    ML_linesearch_BFGS_real_efficientv2(current_estimate, current_gradient, pk, t1, t2, t3, A, Ad, Lh1, Lh2, Lg1, var, N, L, s1_matrix, E, F, y, zk_matrix, t, LScountVec(SBcount-1)) ;
    sk = t*pk ;
    current_estimate = current_estimate+t*(pk(arma::span(0, LT-1))+cx_double(0, 1)*pk(arma::span(LT, 2*LT-1))) ;
    previous_gradient = current_gradient ;
    current_gradient = obtain_gradient_real_efficientv8(current_estimate, t1, t2, t3, A, Ad, Lh1, Lh2, Lg1, var, N, L, s1_matrix, E, F, y, zk_matrix) ;
    yk = current_gradient-previous_gradient ;
    rhok = 1/(arma::as_scalar(arma::trans(yk)*sk)) ;
    if (arma::as_scalar(arma::trans(yk)*sk)>pow(10, (-8)))
    {
      Hk = (arma::eye<mat>(2*LT, 2*LT)-rhok*sk*arma::trans(yk))*Hk*(arma::eye<mat>(2*LT, 2*LT)-rhok*yk*arma::trans(sk))+rhok*sk*arma::trans(sk) ;
    }
  }
  LScountVec = LScountVec(arma::span(0, SBcount-1)) ;
  LScount = mean(LScountVec) ;
  BFGS_estimate = current_estimate ;
}

vec obtain_gradient_real_efficientv8(cx_mat estimate, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat F, cx_mat y, cx_mat zk_matrix)
{
  cx_mat C1inv, G1, H1, H2, Pgradienti, Q1inv, dQ1inv_g1i, dQ1inv_h2i, dmu1_g1iF, dmu1_h1i, dmu1_h2iF, dmu2_h1i_matrix, dmu2_h2i_matrix, g1_app, g1_fft, g1_ffti, g1c_fft, h1_app, h2_app, h2_fft, h2_ffti, h2c_fft, h2c_ffti, mu1, mu2_matrix, v1F, v3i, v4i, ymu1F, zkmu2_matrix ;
  int i ;
  vec SB_gradient, real_gradient, v1, v2 ;
  h1_app = arma::join_cols(estimate(arma::span(0, Lh1-1)), arma::zeros<cx_mat>(N-Lh1, 1)) ;
  h2_app = arma::join_cols(estimate(arma::span(Lh1, Lh1+Lh2-1)), arma::zeros<cx_mat>(N-Lh2, 1)) ;
  g1_app = arma::join_cols(estimate(arma::span(Lh1+Lh2, Lh1+Lh2+Lg1-1)), arma::zeros<cx_mat>(N-Lg1, 1)) ;

  rotate_matrix(size(h1),&h2_app, &H2);
  rotate_matrix(size(h1),&G1_app, &G1);
  h2_fft = arma::fft(h2_app) ;
  h2c_fft = arma::conj(h2_fft) ;
  g1_fft = arma::fft(g1_app) ;
  g1c_fft = arma::conj(g1_fft) ;
  v1 = 1/(var*(pow(A, 2)*arma::square(abs(h2_fft))+1)) ;
  C1inv = F*diagmat(v1)*arma::trans(F) ;
  v2 = 1/(pow(Ad, 2)*E*arma::square(abs(h2_fft%g1_fft))+pow(Ad, 2)*var*(arma::square(abs(h2_fft)))+var) ;
  Q1inv = F*diagmat(v2)*arma::trans(F) ;
  mu2_matrix = Ad*H2*H1*s1_matrix ;
  zkmu2_matrix = zk_matrix-mu2_matrix ;
  mu1 = A*H2*H1*t1+A*H2*G1*t2+H2*t3 ;
  ymu1F = arma::trans(F)*(y-mu1) ;
  SB_gradient = arma::zeros<vec>(Lh1+Lh2+Lg1) ;
  for (i=1; i<=Lh1+Lh2+Lg1; i++)
  {
    if (i<=Lh1)
    {
      dmu1_h1i = A*H2.cols(arma::strans(arma::join_rows(m2cpp::span<rowvec>(i, N), m2cpp::span<rowvec>(1, i-1)))-1)*t1 ;
      Pgradienti = -arma::trans((y-mu1))*C1inv*dmu1_h1i ;
      dmu2_h1i_matrix = Ad*H2.cols(arma::strans(arma::join_rows(m2cpp::span<rowvec>(i, N), m2cpp::span<rowvec>(1, i-1)))-1)*s1_matrix ;
      SB_gradient(i-1) = Pgradienti-trace(arma::trans(zkmu2_matrix)*Q1inv*dmu2_h1i_matrix) ;
    }
    else if (i<=Lh1+Lh2)
    {
      dmu1_h2iF = arma::trans(F)*(A*H1.cols(arma::strans(arma::join_rows(m2cpp::span<rowvec>(i-Lh1, N), m2cpp::span<rowvec>(1, i-Lh1-1)))-1)*t1+A*G1.cols(arma::strans(arma::join_rows(m2cpp::span<rowvec>(i-Lh1, N), m2cpp::span<rowvec>(1, i-Lh1-1)))-1)*t2+t3(arma::strans(arma::join_rows(m2cpp::span<rowvec>(N-i+Lh1+2, N), m2cpp::span<rowvec>(1, N-i+Lh1+1)))-1)) ;
      h2c_ffti = h2c_fft%arma::exp(-cx_double(0, 1)*2*datum::pi*(i-Lh1-1)*arma::strans((m2cpp::fspan(0, 1, N-1)))/N) ;
      g1_ffti = g1_fft%arma::exp(-cx_double(0, 1)*2*datum::pi*(i-Lh1-1)*arma::strans((m2cpp::fspan(0, 1, N-1)))/N) ;
      v3i = pow(Ad, 2)*E*g1_ffti%g1c_fft%h2c_fft+pow(Ad, 2)*var*h2c_ffti ;
      dQ1inv_h2i = -F*diagmat(arma::square(v2)%v3i)*arma::trans(F) ;
      v1F = var*pow(A, 2)*v1%h2c_ffti ;
      Pgradienti = arma::sum(v1F)-arma::sum(ymu1F%v1%v1F%arma::conj(ymu1F))-arma::sum(arma::conj(ymu1F)%v1%dmu1_h2iF) ;
      dmu2_h2i_matrix = Ad*H1.cols(arma::strans(arma::join_rows(m2cpp::span<rowvec>(i-Lh1, N), m2cpp::span<rowvec>(1, i-Lh1-1)))-1)*s1_matrix ;
      SB_gradient(i-1) = Pgradienti+trace(arma::trans(zkmu2_matrix)*dQ1inv_h2i*zkmu2_matrix)+L*arma::sum(v2%v3i)-trace(arma::trans(zkmu2_matrix)*Q1inv*dmu2_h2i_matrix) ;
    }
    else
    {
      dmu1_g1iF = arma::trans(F)*A*H2.cols(arma::strans(arma::join_rows(m2cpp::span<rowvec>(i-Lh1-Lh2, N), m2cpp::span<rowvec>(1, i-Lh1-Lh2-1)))-1)*t2 ;
      Pgradienti = -arma::sum(arma::conj(ymu1F)%v1%dmu1_g1iF) ;
      h2_ffti = h2_fft%arma::exp(-cx_double(0, 1)*2*datum::pi*(i-Lh1-Lh2-1)*arma::strans((m2cpp::fspan(0, 1, N-1)))/N) ;
      v4i = pow(Ad, 2)*E*h2_ffti%g1c_fft%h2c_fft ;
      dQ1inv_g1i = -F*diagmat(arma::square(v2)%v4i)*arma::trans(F) ;
      SB_gradient(i-1) = Pgradienti+L*arma::sum(v2%v4i)+trace(arma::trans(zkmu2_matrix)*dQ1inv_g1i*zkmu2_matrix) ;
    }
  }
  real_gradient = {2*arma::real(SB_gradient), -2*arma::imag(SB_gradient)} ;
  return real_gradient ;
}

void ML_linesearch_BFGS_real_efficientv2(cx_mat estimate, vec gradient, vec delta, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat F, cx_mat y, cx_mat zk_matrix, int& t, int& count)
{
  double alpha, beta ;
  float ML_original ;
  int LT ;
  alpha = pow(10, (-4)) ;
  beta = 0.9 ;
  t = 1 ;
  LT = Lh1+Lh2+Lg1 ;
  count = 1 ;
  ML_original = ML_eval_SB_non_new_efficient(estimate, t1, t2, t3, A, Ad, Lh1, Lh2, Lg1, var, N, L, s1_matrix, E, F, y, zk_matrix) ;
  while (ML_eval_SB_non_new_efficient(estimate+t*(delta(arma::span(0, LT-1))+cx_double(0, 1)*delta(arma::span(LT, 2*LT-1))), t1, t2, t3, A, Ad, Lh1, Lh2, Lg1, var, N, L, s1_matrix, E, F, y, zk_matrix)>ML_original+arma::as_scalar(alpha*t*arma::trans(gradient)*delta))
  {
    count = count+1 ;
    t = beta*t ;
  }
}

float ML_eval_SB_non_new_efficient(cx_mat estimate, cx_mat t1, cx_mat t2, cx_mat t3, float A, float Ad, int Lh1, int Lh2, int Lg1, float var, int N, int L, cx_mat s1_matrix, float E, cx_mat F, cx_mat y, cx_mat zk_matrix)
{
  cx_mat G1, H1, H2, g1_app, g1_fft, h1_app, h2_app, h2_fft, mu1, mu2_matrix, ymu1F, zkmuF1 ;
  float ML_value, ML_value1, ML_value2 ;
  vec v1, v2 ;
  h1_app = arma::join_cols(estimate(arma::span(0, Lh1-1)), arma::zeros<cx_mat>(N-Lh1, 1)) ;
  h2_app = arma::join_cols(estimate(arma::span(Lh1, Lh1+Lh2-1)), arma::zeros<cx_mat>(N-Lh2, 1)) ;
  g1_app = arma::join_cols(estimate(arma::span(Lh1+Lh2, Lh1+Lh2+Lg1-1)), arma::zeros<cx_mat>(N-Lg1, 1)) ;
  rotate_matrix(size(h1),&h1_app, &H1);
  rotate_matrix(size(h1),&h2_app, &H2);
  rotate_matrix(size(h1),&G1_app, &G1);
  h2_fft = arma::fft(h2_app) ;
  g1_fft = arma::fft(g1_app) ;
  v1 = 1/(var*(pow(A, 2)*arma::square(abs(h2_fft))+1)) ;
  v2 = 1/(pow(Ad, 2)*E*arma::square(abs(h2_fft%g1_fft))+pow(Ad, 2)*var*(arma::square(abs(h2_fft)))+var) ;
  mu1 = A*H2*(H1*t1+G1*t2)+H2*t3 ;
  ymu1F = arma::trans(F)*(y-mu1) ;
  ML_value1 = -arma::log(prod(v1))+arma::sum(v1%ymu1F%arma::conj(ymu1F)) ;
  mu2_matrix = Ad*H2*H1*s1_matrix ;
  zkmuF1 = arma::trans(F)*(zk_matrix-mu2_matrix) ;
  ML_value2 = arma::sum(arma::sum(repmat(v2, 1, L)%zkmuF1%arma::conj(zkmuF1))) ;
  ML_value = arma::real(ML_value1+ML_value2-L*arma::log(prod(v2))) ;
  return ML_value ;
}

mat generate_IDFT(int L)
{
  int m, n ;
  mat F ;
  F = arma::zeros<mat>(L, L) ;
  for (m=1; m<=L; m++)
  {
    for (n=1; n<=L; n++)
    {
      F(m-1, n-1) = 1*1.0/sqrt(L)*arma::exp(cx_double(0, 1)*2*datum::pi*(m-1)*(n-1)*1.0/L) ;
    }
  }
  return F ;
}
void rotate_matrix(int sizeofvector, cx_mat *in_vec, cx_mat *out_mat)
{
for( int j = 0; j < sizeofvector; j++)
{
  for( int i = 0; i < sizeofvector; i++)
{
    if(j == 0)
        out_mat(i,j) = in_vec[i];
    else
    {
        if(i == 0)
            {
                out_mat(i,j) = temp;
                i++;
            }
        out_mat[i][j] = out_mat(i-1,j-1);
    }
if (i == 3)
        temp = out_mat(i,j);
}
}
}
