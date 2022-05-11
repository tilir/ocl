#!/usr/bin/ruby

#------------------------------------------------------------------------------
#
# Running some numerical experiments 
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

puts "Running SGEMMs";

def run_sgemm(progname, outfile, ax, ax0, ay, by, lsz)
  puts "Running for lsz = #{lsz}, ax = #{ax}"
  outmark = ">>"
  outmark = ">" if ax == ax0
  sysline = "#{progname} -q=1 -ay=#{ay} -by=#{by} -ax=#{ax} -lsz=#{lsz} #{outmark} #{outfile}"
  puts("#{sysline}")
  system("#{sysline}")
end

device = "tgllp"
ax0 = 4
axn = 20
ay = 8
by = 6
lsz = 8

progname = "sgemm\\matmult_nopriv.exe"

outfile = "gemm_nopriv.dat"
(ax0..axn).each { |ax| run_sgemm(progname, outfile, ax, ax0, ay, by, lsz) }

progname = "sgemm\\matmult.exe"

outfile = "gemm_priv.dat"
(ax0..axn).each { |ax| run_sgemm(progname, outfile, ax, ax0, ay, by, lsz) }

progname = "sgemm\\matmult_local.exe"

outfile = "gemm_lsz#{lsz}.dat"
(ax0..axn).each { |ax| run_sgemm(progname, outfile, ax, ax0, ay, by, lsz) }

lsz = 16
outfile = "gemm_lsz#{lsz}.dat"
(ax0..axn).each { |ax| run_sgemm(progname, outfile, ax, ax0, ay, by, lsz) }
