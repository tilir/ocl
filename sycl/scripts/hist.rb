#!/usr/bin/ruby

#------------------------------------------------------------------------------
#
# Runner for numerical experiments on matrix histogramm
# see hist.plot on how to collect data
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
options.progname = "histogram\\hist_naive.exe"
options.outfile = "hist.dat"
options.lsz = 256
options.bsz = 1024
options.szmult = 10000
options.sz0 = 10
options.szn = 20
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

def run_hist(progname, outfile, szx, sz0, szmult, lsz, bsz)
  sz = szx * szmult
  puts "Running for lsz = #{lsz}, sz = #{sz}"
  outmark = ">>"
  outmark = ">" if szx == sz0
  sysline = "#{progname} -quiet=1 -sz=#{sz} -lsz=#{lsz} -bsz=#{bsz} #{outmark} #{outfile}"
  puts("#{sysline}")
  system("#{sysline}")
end

sz0 = options.sz0
szn = options.szn

(sz0..szn).each do |sz|
  run_hist(options.progname, options.outfile, sz, sz0, options.szmult, options.lsz, options.bsz)
end
