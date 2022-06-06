#------------------------------------------------------------------------------
#
# Gnuplot script for bitonic experiments
#
# collect data with bitonic.rb
# ..\scripts\bitonic.rb -p bitonic\bitonic_buffer.exe -o bitonic_buffer.dat
# ..\scripts\bitonic.rb -p bitonic\bitonic_device.exe -o bitonic_device.dat
# ..\scripts\bitonic.rb -p bitonic\bitonic_device_local.exe -o bitonic_device_local.dat
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
plot 'bitonic_buffer.dat' with linespoints t 'Accessor',\
     'bitonic_device.dat' with linespoints t 'Device memory',\
     'bitonic_device_local.dat' with linespoints t 'Local memory'   