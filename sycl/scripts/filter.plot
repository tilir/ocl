#------------------------------------------------------------------------------
#
# Gnuplot script for filter experiments
#
# collect data with filter.rb
# ..\scripts\filter.rb -p filtering\filtering_sampler.exe -o filtering_sampler.dat
# ..\scripts\filter.rb -p filtering\filtering_sampler_nonvectorized.exe -o filtering_sampler_nonvectorized.dat
# ..\scripts\filter.rb -p filtering\filtering_buffer.exe -o filtering_buffer.dat
# ..\scripts\filter.rb -p filtering\filtering_shared.exe -o filtering_shared.dat
# ..\scripts\filter.rb -p filtering\filtering_sampler_local.exe -o filtering_sampler_local.dat
# ..\scripts\filter.rb -p filtering\filtering_buffer_local.exe -o filtering_buffer_local.dat
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
set xlabel "Image width (filter size = 9)"
set ylabel "Time (seconds)"

set output "samplers_vectorization.png"
plot 'filtering_sampler_nonvectorized.dat' with linespoints t 'Non-vectorized sampler',\
     'filtering_sampler.dat' with linespoints t 'Baseline sampler'
     
set output "samplers_buffers.png"
plot 'filtering_sampler.dat' with linespoints t 'Baseline sampler',\
     'filtering_buffer.dat' with linespoints t 'Filtering with buffers',\
     'filtering_shared.dat' with linespoints t 'Filtering with shared memory'
     
set output "samplers_locals.png"
plot 'filtering_sampler.dat' with linespoints t 'Baseline sampler',\
     'filtering_sampler_local.dat' with linespoints t 'Baseline sampler with local memory'

set output "samplers_buffers_local.png"
plot 'filtering_sampler.dat' with linespoints t 'Baseline sampler',\
     'filtering_buffer_local.dat' with linespoints t 'Buffers with local memory',\
     'filtering_sampler_local.dat' with linespoints t 'Baseline sampler with local memory'
