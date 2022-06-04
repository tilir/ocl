#!/usr/bin/ruby

#------------------------------------------------------------------------------
#
# Runner for numerical experiments on convolutions
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

puts "Running bitonic sorts";

ARGV << '-h' if ARGV.empty?

options = OpenStruct.new
options.verbose = false
options.progname = "filtering\\filtering_sampler.exe"
options.outfile = "filtering_sampler.dat"
options.rbsz = 256
options.lsz = 16
options.sz0 = 7
options.step = 10
options.npoints = 10
options.device = "tgllp"

OptionParser.new do |opts|
  opts.banner = "Usage: #{$0} -p <progname> -o <outfile> [other opts]"
  opts.on("-v", "--[no-]verbose", "Run verbosely (default: #{options.verbose})") { |v| options.verbose = v }
  opts.on("-p", "--progname p", "Program name to run (default: #{options.progname})") { |v| options.progname = v }
  opts.on("-o", "--outfile o", "Output file (default: #{options.outfile})") { |v| options.outfile = v }
  opts.on("-l", "--local-size l", Integer, "Local memory size (default: #{options.lsz})") { |v| options.lsz = v }
  opts.on("-i", "--image-size i", Integer, "Image size (default: #{options.rbsz})") { |v| options.rbsz = v }
  opts.on("-f", "--first f", Integer, "Starting filter size (default: #{options.sz0})") { |v| options.sz0 = v }
  opts.on("-s", "--step s", Integer, "Filter size step (default: #{options.step})") { |v| options.step = v }
  opts.on("-n", "--npoints n", Integer, "Number of points (default: #{options.npoints})") { |v| options.npoints = v }
  opts.on("-d", "--devname f", String, "Device name (default: #{options.device})") { |v| options.device = v }
  opts.on_tail("-h", "--help", "Show this message") do
    puts opts
    exit
  end
end.parse!

def run_filter(progname, outfile, n, step, sz0, rbsz, lsz)
  sz = sz0 + step * n;
  puts "Running for rbsz = #{rbsz}, lsz = #{lsz}, size = #{sz}"
  outmark = ">>"
  outmark = ">" if n == 0
  sysline = "#{progname} -quiet -randboxes=#{rbsz} -randfilter=#{sz} -lsz=#{lsz} #{outmark} #{outfile}"
  puts("#{sysline}")
  system("#{sysline}")
end

device = "tgllp"
sz0 = options.sz0
step = options.step
lsz = options.lsz
rbsz = options.rbsz
npoints = options.npoints

# run bitonic sort
(0..npoints).each { |x| run_filter(options.progname, options.outfile, x, step, sz0, rbsz, lsz) }