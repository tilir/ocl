#!/usr/bin/ruby

#------------------------------------------------------------------------------
#
# Running some numerical experiments on bitonic sorts
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

puts "Running bitonic sorts";

def run_bsort(progname, outfile, sz, sz0, lsz)
  puts "Running for lsz = #{lsz}, size = #{sz}"
  outmark = ">>"
  outmark = ">" if sz == sz0
  sysline = "#{progname} -q=1 -size=#{sz} -lsz=#{lsz} #{outmark} #{outfile}"
  puts("#{sysline}")
  system("#{sysline}")
end

device = "tgllp"
sz0 = 18
szn = 26
lsz = 1

# bitonic sort with 1D range, no explicit local memory
progname = "bitonic\\bitonic_simple.exe"
outfile = "bitonic_simple.dat"
(sz0..szn).each { |sz| run_bsort(progname, outfile, sz, sz0, lsz) }

# bitonic sort with ND range, explicit local memory = 1
progname = "bitonic\\bitonicsort.exe"
outfile = "bitonic_nd_#{lsz}.dat"
(sz0..szn).each { |sz| run_bsort(progname, outfile, sz, sz0, lsz) }

# bitonic sort with ND range, explicit local memory = 1
progname = "bitonic\\bitonicsort.exe"
lsz = 32
outfile = "bitonic_nd_#{lsz}.dat"
(sz0..szn).each { |sz| run_bsort(progname, outfile, sz, sz0, lsz) }
