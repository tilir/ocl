#------------------------------------------------------------------------------
#
# Gnuplot script for filter experiments
#
# collect data with filter.rb
# ..\scripts\filter.rb -p filtering\filtering_sampler.exe -o filtering_sampler.dat
# ..\scripts\filter.rb -p filtering\filtering_sampler_vectorized.exe -o filtering_sampler_vectorized.dat
# ..\scripts\filter.rb -p filtering\filtering_sampler.exe -o filtering_sampler_n6.dat -n 6
# ..\scripts\filter.rb -p filtering\filtering_buffer.exe -o filtering_buffer_n6.dat -n 6
# ..\scripts\filter.rb -p filtering\filtering_local.exe -o filtering_local.dat
# ..\scripts\filter.rb -p filtering\filtering_sampler_local.exe -o filtering_sampler_local.dat
# ..\scripts\filter.rb -p filtering\filtering_sampler_vectorized.exe -o filtering_sampler_2048.dat -i 2048 -f 3 -s 4
# ..\scripts\filter.rb -p filtering\filtering_sampler_local.exe -o filtering_sampler_local_2048.dat -i 2048 -f 3 -s 4
#
# run plotter with
# > gnuplot -persist -c ..\scripts\filter.plot
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
set xlabel "Filter size"
set ylabel "time (seconds)"

set output "samplers_vectorization.png"
plot 'filtering_sampler.dat' with linespoints t 'Non-vectorized sampler',\
     'filtering_sampler_vectorized.dat' with linespoints t 'Vectorized sampler'
     
set output "samplers_buffers.png"
plot 'filtering_sampler_n6.dat' with linespoints t 'Non-vectorized sampler',\
     'filtering_buffer_n6.dat' with linespoints t 'Non-vectorized buffer'
     
set output "samplers_locals.png"
plot 'filtering_sampler_2048.dat' with linespoints t 'Vectorized sampler',\
     'filtering_sampler_local_2048.dat' with linespoints t 'Vectorized sampler with 16x16 local memory'
