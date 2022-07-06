#------------------------------------------------------------------------------
#
# Gnuplot script for matrix experiment
#
# private memory used/not used without local memory
#
# collect data with gemm.rb
# ..\scripts\gemm.rb -p sgemm\matmult_mkl.exe -o gemm_mkl.dat
# ..\scripts\gemm.rb -p sgemm\matmult_local_shared.exe -l 8 -o gemm_lsz8.dat
# ..\scripts\gemm.rb -p sgemm\matmult_local_shared.exe -l 16 -o gemm_lsz16.dat
# ..\scripts\gemm.rb -p sgemm\matmult_local_shared_spec.exe -l 16 -o gemm_lsz16_spec.dat
# ..\scripts\gemm.rb -p sgemm\matmult_local_shared_nobundle.exe -l 16 -o gemm_lsz16_nobundle.dat
# ..\scripts\gemm.rb -p sgemm\matmult_groups.exe -l 16 -o gemm_groups_lsz16.dat
# ..\scripts\gemm.rb -p sgemm\matmult_groups_priv.exe -l 16 -o gemm_groups_priv_lsz16.dat
#
# run plotter with
# > gnuplot -persist -c ..\scripts\gemm_local.plot
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

# MKL vs explicit groups (local size = 16)
set output "sgemm_groups.png"
plot 'gemm_mkl.dat' with linespoints title 'MKL baseline',\
     'gemm_groups_lsz16.dat' with linespoints t 'Explicit groups (no priv mem)',\
     'gemm_groups_priv_lsz16.dat' with linespoints t 'Explicit groups'

# MKL vs local (different sizes)
set output "sgemm_lsz.png"
plot 'gemm_mkl.dat' with linespoints title 'MKL baseline',\
     'gemm_lsz8.dat' with linespoints t 'Local memory 8x8',\
     'gemm_lsz16.dat' with linespoints t 'Local memory 16x16'

# Specialization constant effect
set output "sgemm_lsz_spec.png"
plot 'gemm_lsz16.dat' with linespoints t 'Local memory 16x16',\
     'gemm_lsz16_spec.dat' with linespoints t 'Local memory 16x16, spec const, bundles',\
     'gemm_lsz16_nobundle.dat' with linespoints t 'Local memory 16x16, spec const'