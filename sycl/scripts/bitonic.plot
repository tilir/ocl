#------------------------------------------------------------------------------
#
# Gnuplot script for bitonic experiments
#
# collect data with bitonic.rb
# ..\scripts\bitonic.rb -p bitonic\bitonicsort.exe -o bitonicsort.dat
# ..\scripts\bitonic.rb -p bitonic\bitonic_shared.exe -o bitonic_shared.dat
#
# run plotter with
# > gnuplot -persist -c ..\scripts\bitonic.plot
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
set xlabel "Array size (logarithmic)"
set ylabel "time (seconds)"

set output "bitonic_baseline.png"
plot 'bitonicsort.dat' with linespoints t 'Accessor',\
     'bitonic_shared.dat' with linespoints t 'Shared memory',
     
     