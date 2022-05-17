#------------------------------------------------------------------------------
#
# Gnuplot script for bitonic experiments
#
# collect data with bitonic.rb
#
# run plotter with
# > gnuplot -c bitonic.plot
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

# set term pdf
set grid
set key left top
set xlabel "Array size (logarithmic)"
set ylabel "time (seconds)"

# 1D vs ND vs ND+local
# set output "bitonic_nd.pdf"
plot 'bitonic_simple.dat' with linespoints t '1D range',\
     'bitonic_nd_1.dat' with linespoints title 'ND range, local memory = 1',\
     'bitonic_nd_32.dat' with linespoints title 'ND range, local memory = 32'