#------------------------------------------------------------------------------
#
# Gnuplot script for histogram experiment
#
# collect data with hist.rb
# ..\scripts\reduction.rb -p reduction\reduce_naive.exe -o reduce_naive.dat
# ..\scripts\reduction.rb -p reduction\reduce_object.exe -o reduce_object.dat
#
# run plotter with
# > gnuplot -persist -c ..\scripts\reduction.plot
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
set xlabel "Data size (given BSZ = 1024, LSZ=256)"
set ylabel "Time (seconds)"

set output "reduce.png"
plot 'reduce_naive.dat' with linespoints title 'Reduction baseline',\
     'reduce_object.dat' with linespoints title 'Reduction object'
