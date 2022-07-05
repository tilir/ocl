#------------------------------------------------------------------------------
#
# Gnuplot script for matrix experiment
#
# private memory used/not used without local memory
#
# collect data with gemm.rb
# ..\scripts\gemm.rb -p sgemm\matmult.exe -o gemm_priv.dat
# ..\scripts\gemm.rb -p sgemm\matmult_device.exe -o gemm_priv_device.dat
# ..\scripts\gemm.rb -p sgemm\matmult_shared.exe -o gemm_priv_shared.dat
# ..\scripts\gemm.rb -p sgemm\matmult_usm.exe -o gemm_priv_usm.dat
# ..\scripts\gemm.rb -p sgemm\matmult_nopriv.exe -o gemm_nopriv.dat
# ..\scripts\gemm.rb -p sgemm\matmult_transposed.exe -o gemm_trans.dat
# ..\scripts\gemm.rb -p sgemm\matmult_mkl.exe -o gemm_mkl.dat
# ..\scripts\gemm.rb -p sgemm\matmult_mkl_trans.exe -o gemm_mkl_trans.dat
# ..\scripts\gemm.rb -p sgemm\matmult_specialization.exe -o gemm_spec.dat
# ..\scripts\gemm.rb -p sgemm\matmult_specialization_svm.exe -o gemm_spec_svm.dat
#
# run plotter with
# > gnuplot -persist -c ..\scripts\gemm_priv.plot
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

set term png
set grid
set key left top
set xlabel "AX size (given AY is 2048 and BY is 1536)"
set ylabel "time (seconds)"

# private vs non-private vs MKL
set output "sgemm_priv.png"
plot 'gemm_nopriv.dat' with linespoints t 'Private memory not used',\
     'gemm_priv.dat' with linespoints title 'Accumulator in private memory',\
     'gemm_mkl.dat' with linespoints t 'MKL multiplication'

# effects of buffer vs device vs shared
set output "sgemm_buf.png"
plot 'gemm_priv.dat' with linespoints t 'Buffers',\
     'gemm_priv_device.dat' with linespoints title 'Device memory',\
     'gemm_priv_shared.dat' with linespoints title 'Shared memory',\
     'gemm_priv_usm.dat' with linespoints title 'Unified memory allocator',\
     'gemm_mkl.dat' with linespoints t 'MKL multiplication'

# effect of specialization constant
set output "sgemm_spec.png"
plot 'gemm_spec.dat' with linespoints t 'Specialization constant',\
     'gemm_spec_svm.dat' with linespoints t 'Specialization constants with SVM',\
     'gemm_mkl.dat' with linespoints t 'MKL multiplication'

# effects of transpositions
set output "sgemm_trans.png"
plot 'gemm_trans.dat' with linespoints t 'SVM multiplication with transpose',\
     'gemm_mkl_trans.dat' with linespoints title 'MKL multiplication with transpose',\
     'gemm_mkl.dat' with linespoints t 'MKL multiplication'