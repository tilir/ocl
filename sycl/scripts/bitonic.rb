#!/usr/bin/ruby

#------------------------------------------------------------------------------
#
# Runner for numerical experiments on bitonic sorts
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
options.progname = "bitonic\\bitonicsort.exe"
options.outfile = "bitonicsort.dat"
options.lsz = 256
options.sz0 = 18
options.szn = 26
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

def run_bsort(progname, outfile, sz, sz0, lsz)
  puts "Running for lsz = #{lsz}, size = #{sz}"
  outmark = ">>"
  outmark = ">" if sz == sz0
  sysline = "#{progname} -quiet=1 -size=#{sz} -lsz=#{lsz} #{outmark} #{outfile}"
  puts("#{sysline}")
  system("#{sysline}")
end

device = "tgllp"
sz0 = options.sz0
szn = options.szn
lsz = options.lsz

# run bitonic sort
(sz0..szn).each { |sz| run_bsort(options.progname, options.outfile, sz, sz0, lsz) }