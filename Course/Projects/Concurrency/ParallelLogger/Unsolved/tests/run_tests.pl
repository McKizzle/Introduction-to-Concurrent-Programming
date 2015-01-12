#!/usr/bin/env perl
use strict;
use warnings;
use Data::Dumper;
use Cwd;
#use threads;

my $path = cwd();
my $bin_dirpath = shift or die "Usage: ./run_tests.pl <bin_dirpath> [client spawn count] [log_file] [run time milliseconds]";
my $client_count;
$client_count = shift or $client_count = 20;    # default to 20 clients. 
my $log_file;
$log_file = shift or $log_file = "testing.log"; # default to testing.log
my $run_time_ms;
$run_time_ms = shift or $run_time_ms = 5000;    # default to five seconds.
my $test_results_file = "test_results.txt";     # output the testing results to this file. 

my @fork_pids = ();
my @client_pids = (); 
my $server_pid = undef;
my $psid = undef;

# spawn a child to run the server. 
if($psid = fork()) { 
    push @fork_pids, $psid;
} else {
    my $out = system("$bin_dirpath/server $log_file &");
}

# Spawn $client_count children to run client instances.  
for(1...$client_count) {
    if(!$psid) { last; } # double last if we are inside of a child process. 
    if($psid = fork()) {
        push @fork_pids, $psid;
    } else {
        my $out = system("$bin_dirpath/client &");
        last;
    }
}


# Now terminate the server and the clients. 
if($psid) {
    print "Letting log build for " . ($run_time_ms / 1000.0) . " seconds.\n";
    sleep $run_time_ms / 1000.0;

    $SIG{CHLD} = sub { wait }; # Avoid the zombie apocalypse on your machine.  

    my $killall_out = `killall -e server -s SIGINT && killall -e client -s SIGINT`;
    my $rm_out      = `rm -f com[0-9]*fifo CommNameSender RequestConnection`;
}

# Parent will now check the log file for inconsistancies.  
my $incosistency_found = 0;
if($psid) {
    # Test the log file for race conditions. 
    open my $file, '<', $log_file;
    open my $result_file, '>', $test_results_file;
    
    my $expected_log_number = 1;
    my $line_number = 1;
    while(<$file>) {
        if($_ =~ /^([0-9]+):\sLog\smessage\sfrom\sprocess\s([0-9]+)[.]\s*$/) { 
            if($1 ne $expected_log_number) { 
                print $result_file "Inconsistency found on line: $line_number\n"; 
                print $result_file "\tExpected log number $expected_log_number got $1 instead for process $2's message.\n";
                $expected_log_number = $1;
                $incosistency_found = 1;
            }
        } 
        else {
            print $result_file "Inconsistency found on line: $line_number\n"; 
            print $result_file "\t$_\n";
            $incosistency_found = 1;
        }
        $line_number++;
        $expected_log_number++;
    }
    print "\n";
   
    close $result_file;
    close $file;

    if($line_number eq 1) {
        $incosistency_found = 1;
    }
}

exit $incosistency_found;



