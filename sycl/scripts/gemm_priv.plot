#------------------------------------------------------------------------------
#
# Gnuplot script for matrix experiment
#
# private memory used/not used without local memory
# local memory used for sizes 8 and 16
#
# collect data with gemm_priv.rb
#
# run plotter with
# > gnuplot -persist -c gemm_priv.plot
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

set term pdf
set grid
set key left top
set xlabel "AX size (given AY is 2048 and BY is 1536)"
set ylabel "time (seconds)"

# private vs non-prviate
set output "sgemm_priv.pdf"
plot 'gemm_nopriv.dat' with linespoints t 'Private memory not used',\
     'gemm_priv.dat' with linespoints title 'Accumulator in private memory'

# private vs local (different sizes)
set output "sgemm.pdf"
plot 'gemm_priv.dat' with linespoints title 'Accumulator in private memory',\
     'gemm_lsz8.dat' with linespoints t 'Local memory 8x8',\
     'gemm_lsz16.dat' with linespoints t 'Local memory 16x16'

# private vs local (different sizes) vs MKL
set output "sgemm_mkl.pdf"
plot 'gemm_nopriv.dat' with linespoints t 'Private memory not used',\
     'gemm_priv.dat' with linespoints title 'Accumulator in private memory',\
     'gemm_mkl.dat' with linespoints t 'MKL multiplication',\
     'gemm_mkl_trans.dat' with linespoints t 'MKL multiplication with transpose',\
     'gemm_lsz8.dat' with linespoints t 'Local memory 8x8',\
     'gemm_lsz16.dat' with linespoints t 'Local memory 16x16'