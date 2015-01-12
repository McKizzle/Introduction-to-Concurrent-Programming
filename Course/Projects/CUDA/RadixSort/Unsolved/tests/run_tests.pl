#!/usr/bin/env perl 
use strict;
use warnings;
use Data::Dumper;
use Cwd; 

my $bin_path = shift; #cwd();
my $shuffled_path = shift or die usage();
my $sorted_path = shift or die usage();
my $golden_path = shift or die usage();
$/ = shift or die usage(); 

mkdir $sorted_path;

opendir(my $shuffled_dir, $shuffled_path) or die "Unable to open $shuffled_path. Did you generate any test vectors?\n If not use the vecGen.py python script.\n";

# run radix_sort on the shuffled vectors and compare its output to the expected vectors. 
my $error = 0;
while(my $file = readdir($shuffled_dir)) {
    if(-f "$shuffled_path/$file" && ($file =~ /(\d+)[.]vec$/ )) {
        `cat "$shuffled_path/$file" | $bin_path $/ $1 - > $sorted_path/$1.vec`;
        my $sorted = file4vector("$sorted_path/$1.vec");
        my $golden = file4vector("$golden_path/$1.vec");
        my $miss_index = compare_vectors($sorted, $golden);
        if(!defined($miss_index)) {
            print "sorted/$1.vec & golden/$1.vec are not the same length. Did you complete the program or did you pass in the incorrect delimeter?\n";
        } elsif( $miss_index < 0 ) {
            print "sorted/$1.vec & golden/$1.vec match up.\n";
        }
        else {
            $error = 1;
            print "sorted/$1.vec & golden/$1.vec do not match at index $miss_index.\n";
        }
    }
}

exit $error;


# \brief read a file to a vector. 
#
# \param path to the file
#
# \return a list or undef. 
sub file4vector {
    my $filepath = shift;

    open( my $file, '<', $filepath ) or warn "Failed to open $filepath\n";
    
    my @vector;
    while( my $num = <$file> ) {
        chomp $num;
        if($num =~ /[0-9]+/) {
            push @vector, $num;
        }
    }

    close $file;

    return \@vector;
}

# \brief Compares two vectors. 
#
# \param vector ref 1
# \param vector ref 2
#
# \param the index at which the first mismatch occurs. Otherwise return nothing.  
sub compare_vectors {
    my $refvec1 = shift;
    my $refvec2 = shift;
    
    if (@$refvec1 != @$refvec2) {
        warn "The vectors are not the same length\n";    
        return undef;
    }
    
    for(my $i = 0; $i < @$refvec1; $i++) {
        if($refvec1->[$i] != $refvec2->[$i]) {
            return $i;
        }
    }

    return -1;
}

# \brief Return program usage. 
sub usage {
    return "Usage: ./run_tests <radix sort bin path> <shuffled dir path> <sorted dir path> <golden dir path> <delimiter>\n"; 
}


