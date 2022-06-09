#------------------------------------------------------------------------------
#
# Gnuplot script for histogram experiment
#
# collect data with hist.rb
# ..\scripts\hist.rb -p histogram\hist_naive_acc.exe -o hist_naive_acc.dat
# ..\scripts\hist.rb -p histogram\hist_naive.exe -o hist_naive.dat
# ..\scripts\hist.rb -p histogram\hist_local_acc.exe -o hist_local_acc.dat
# ..\scripts\hist.rb -p histogram\hist_local_acc_spec.exe -o hist_local_acc_spec.dat
# ..\scripts\hist.rb -p histogram\hist_local.exe -o hist_local.dat
# ..\scripts\hist.rb -p histogram\hist_private.exe -o hist_private.dat
#
# run plotter with
# > gnuplot -persist -c ..\scripts\hist.plot
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
set xlabel "Data size (given GSZ = 65536, LSZ=256)"
set ylabel "Time (seconds)"

set output "hist_acc.png"
plot 'hist_local_acc.dat' with linespoints title 'Histogram baseline',\
     'hist_local.dat' with linespoints title 'Histogram baseline (SVM)'

# private vs local (different sizes)
set output "hist.png"
plot 'hist_naive_acc.dat' with linespoints title 'Histogram baseline',\
     'hist_local_acc.dat' with linespoints t 'Histogram with local memory',\
     'hist_private.dat' with linespoints t 'Histogram with private memory'

# private vs local (different sizes)
set output "hist_svm.png"
plot 'hist_naive.dat' with linespoints title 'Histogram baseline (SVM)',\
     'hist_local.dat' with linespoints t 'Histogram with local memory (SVM)',\
     'hist_private.dat' with linespoints t 'Histogram with private memory'

# specialization constants
set output "hist_specconst.png"
plot 'hist_local_acc_spec.dat' with linespoints title 'Histogram local memory with specialization constants',\
     'hist_local_acc.dat' with linespoints t 'Histogram local memory'