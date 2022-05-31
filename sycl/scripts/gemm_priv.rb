#!/usr/bin/ruby

#------------------------------------------------------------------------------
#
# Runner for some numerical experiments
#
#------------------------------------------------------------------------------
#
# This file is licensed after LGPL v3
# Look at: https://www.gnu.org/licenses/lgpl-3.0.en.html for details
#
#------------------------------------------------------------------------------

require 'fileutils'
require 'open3'
require 'optparse'
require 'ostruct'

puts "Running SGEMMs";

ARGV << '-h' if ARGV.empty?

options = OpenStruct.new
options.verbose = false
options.progname = "sgemm\\matmult.exe"
options.outfile = "gemm_priv.dat"
options.lsz = 8
options.ax0 = 4
options.axn = 20
options.ay = 8
options.by = 6
options.bsz = 256
options.device = "tgllp"

OptionParser.new do |opts|
  opts.banner = "Usage: #{$0} -p <progname> -o <outfile> [other opts]"
  opts.on("-v", "--[no-]verbose", "Run verbosely (default: #{options.verbose})") { |v| options.verbose = v }
  opts.on("-p", "--progname p", "Program name to run (default: #{options.progname})") { |v| options.progname = v }
  opts.on("-o", "--outfile o", "Output file (default: #{options.outfile})") { |v| options.outfile = v }
  opts.on("-l", "--local-size l", Integer, "Local memory size (default: #{options.lsz})") { |v| options.lsz = v }
  opts.on("-d", "--devname f", String, "Device name (default: #{options.device})") { |v| options.device = v }
  opts.on_tail("-h", "--help", "Show this message") do
    puts opts
    exit
  end
end.parse!

def run_sgemm(progname, outfile, ax, ax0, ay, by, lsz, bsz)
  puts "Running for lsz = #{lsz}, ax = #{ax}"
  outmark = ">>"
  outmark = ">" if ax == ax0
  sysline = "#{progname} -quiet=1 -ay=#{ay} -by=#{by} -ax=#{ax} -lsz=#{lsz} -bsz=#{bsz} #{outmark} #{outfile}"
  puts("#{sysline}")
  system("#{sysline}")
end

ax0 = options.ax0
axn = options.axn

(ax0..axn).each do |ax|
  run_sgemm(options.progname, options.outfile, ax, ax0,
            options.ay, options.by, options.lsz, options.bsz)
end

