cdef extern from *:
    int Py_AtExit(void (*callback)())

cdef extern from "fftw3.h":
    void fftw_cleanup()
    void fftw_init_threads()
    void fftw_plan_with_nthreads(int)
    void fftw_cleanup_threads()

cdef extern from "nfft3.h":
  ctypedef double fftw_complex[2]

  # nfft flags
  ctypedef enum:
    PRE_PHI_HUT                 #1
    FG_PSI                      #2
    PRE_LIN_PSI                 #4
    PRE_FG_PSI                  #8
    PRE_PSI                     #16
    PRE_FULL_PSI                #32
    MALLOC_X                    #64
    MALLOC_F_HAT                #128
    MALLOC_F                    #256
    FFT_OUT_OF_PLACE            #512
    FFTW_INIT                   #1024
    NFFT_SORT_NODES             #2048
    NFFT_OMP_BLOCKWISE_ADJOINT  #4096
    PRE_ONE_PSI #(PRE_LIN_PSI| PRE_FG_PSI| PRE_PSI| PRE_FULL_PSI)

  # fftw flags
  ctypedef enum:
    FFTW_ESTIMATE
    FFTW_DESTROY_INPUT

  # nfft plan
  ctypedef struct nfft_plan:
    fftw_complex * f_hat
    fftw_complex * f
    double *x
    unsigned flags

  # nfft functions
  void nfft_trafo_direct (nfft_plan *ths) nogil
  void nfft_adjoint_direct (nfft_plan *ths) nogil
  void nfft_trafo (nfft_plan *ths) nogil
  void nfft_adjoint (nfft_plan *ths) nogil
  void nfft_init_guru (nfft_plan *ths, int d, int *N, int M, int *n, int m,
                       unsigned nfft_flags, unsigned fftw_flags)
  void nfft_precompute_one_psi (nfft_plan *ths) nogil
  void nfft_finalize (nfft_plan *ths)

  # fpt flags
  ctypedef enum:
    FPT_NO_STABILIZATION      #1
    FPT_NO_FAST_ALGORITHM     #4
    FPT_NO_DIRECT_ALGORITHM   #8
    FPT_PERSISTENT_DATA       #16
    FPT_FUNCTION_VALUES       #32
    FPT_AL_SYMMETRY           #64

  # nfsft flags
  ctypedef enum:
    NFSFT_NORMALIZED            #1
    NFSFT_USE_NDFT              #2
    NFSFT_USE_DPT               #4
    NFSFT_MALLOC_X              #8
    NFSFT_MALLOC_F_HAT          #16
    NFSFT_MALLOC_F              #32
    NFSFT_PRESERVE_F_HAT        #64
    NFSFT_PRESERVE_X            #128
    NFSFT_PRESERVE_F            #256
    NFSFT_DESTROY_F_HAT         #512
    NFSFT_DESTROY_X             #1024
    NFSFT_DESTROY_F             #2048
    NFSFT_NO_DIRECT_ALGORITHM   #4096
    NFSFT_NO_FAST_ALGORITHM     #8192
    NFSFT_ZERO_F_HAT            #16384

  # nfsft plan
  ctypedef struct nfsft_plan:
    fftw_complex * f_hat
    fftw_complex * f
    double *x
    unsigned flags
    int N

  # nfsft functions
  void nfsft_trafo_direct (nfsft_plan *ths) nogil
  void nfsft_adjoint_direct (nfsft_plan *ths) nogil
  void nfsft_trafo (nfsft_plan *ths) nogil
  void nfsft_adjoint (nfsft_plan *ths) nogil
  void nfsft_init_guru (nfsft_plan *ths, int N, int M,
                       unsigned nfsft_flags, unsigned nfft_flags, int nfft_cutoff)
  void nfsft_precompute (int N, double kappa, unsigned nfsft_flags, unsigned fpt_flags) nogil
  void nfsft_precompute_x (nfsft_plan *ths)
  void nfsft_forget ()
  void nfsft_finalize (nfsft_plan *ths)


  # nsoft flags
  ctypedef enum:
    NFSOFT_NORMALIZED         #1
    NFSOFT_USE_NDFT           #2
    NFSOFT_USE_DPT            #4
    NFSOFT_MALLOC_X           #8
    NFSOFT_REPRESENT          #16
    NFSOFT_MALLOC_F_HAT       #32
    NFSOFT_MALLOC_F           #64
    NFSOFT_PRESERVE_F_HAT     #128
    NFSOFT_PRESERVE_X         #256
    NFSOFT_PRESERVE_F         #512
    NFSOFT_DESTROY_F_HAT      #1024
    NFSOFT_DESTROY_X          #2048
    NFSOFT_DESTROY_F          #4096
    NFSOFT_NO_STABILIZATION   #8192
    NFSOFT_CHOOSE_DPT         #16384
    NFSOFT_SOFT               #32768
    NFSOFT_ZERO_F_HAT         #65536

  # nsoft plan
  ctypedef struct nfsoft_plan:
    fftw_complex * f_hat
    fftw_complex * f
    double *x
    unsigned flags

  # nsoft functions
  void 	nfsoft_precompute (nfsoft_plan *plan)
  void 	nfsoft_init (nfsoft_plan *plan, int N, int M)
  void 	nfsoft_init_advanced (nfsoft_plan *plan, int N, int M, unsigned int nfsoft_flags)
  void 	nfsoft_init_guru (nfsoft_plan *plan, int N, int M, unsigned int nfsoft_flags, unsigned int nfft_flags, int nfft_cutoff, int fpt_kappa)
  void 	nfsoft_trafo (nfsoft_plan *plan_nfsoft)
  void 	nfsoft_adjoint (nfsoft_plan *plan_nfsoft)
  void 	nfsoft_finalize (nfsoft_plan *plan)
