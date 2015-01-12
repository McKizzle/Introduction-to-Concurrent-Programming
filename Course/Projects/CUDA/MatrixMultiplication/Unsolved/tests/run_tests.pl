#!/usr/bin/env perl 

use strict;
use warnings;
use Getopt::Long;
use Data::Dumper;
use Pod::Usage;
use Cwd; 

my $numArgs = @ARGV;
my $help = 0;
my $bin_path = undef;
my $matrice_path = undef;
my $golden_path = undef;
my $delimiter = undef;
my $cpu = 0;

GetOptions(
    'help|?|h' => \$help,
    'b|bin-path=s' => \$bin_path,
    'm|matrices-path=s' => \$matrice_path,
    'g|golden-path=s' => \$golden_path,
    'd|delimiter=s' => \$delimiter,
    'c|cpu' => \$cpu
    ) or die "Error parsing arguments\n";

if($help || $numArgs == 0) {
    pod2usage(1);
} else {
    if($delimiter) {
        $delimiter = substr $delimiter, 0, 1;
    } else {
        $delimiter = ' ';
    }
    $/ = $delimiter; 

    opendir my $matrice_dir, $matrice_path or die "Error: $!\n";
    while(my $file = readdir($matrice_dir)) {
        if(-f "$matrice_path/$file" && ($file =~ /([0-9]+)x([0-9]+)-([0-9]+)x([0-9]+).mat/))
        {
            print "------------------[ $file ]----------------------\n";
            my $golden_file = "$1x$2-$3x$4-$1x$4.mat";
            my $result = "";
            if($cpu) {
                print "\tExec: cat $matrice_path/$file | $bin_path -a $1x$2 -b $3x$4 -d \"$delimiter\" --\n";

                $result = `cat $matrice_path/$file | $bin_path -a $1x$2 -b $3x$4 -d "$delimiter" --`;
            } else {
                print "\tExec: cat $matrice_path/$file | $bin_path -a $1x$2 -b $3x$4 -d \"$delimiter\" -g --\n";

                $result = `cat $matrice_path/$file | $bin_path -a $1x$2 -b $3x$4 -d "$delimiter" -g --`;
            }
              
            print "\tGolden File: $golden_path/$golden_file\n";
            my $c_actual = string_for_vector($result);
            my $golden_result = `cat $golden_path/$golden_file`;
            my $c_golden = string_for_vector($golden_result);

            if(@$c_actual ne @$c_golden) {
                print "\tThe program's output size differed from the expected size in $golden_path/$golden_file\n";
                print "\t   Actual: ". scalar @$c_actual ."\n";
                print "\t   Golden: ". scalar @$c_golden ."\n";
            } else {
                my $miss_dims = compare_matrices($c_actual, $c_golden, $1, $4);
                if($miss_dims) { 
                    print "\tThere was a missmatch between the golden (expected) matrix and actual matrix at index:\n";
                    print "\t   row:    $miss_dims->[0]\n";
                    print "\t   column: $miss_dims->[1]\n";
                    print "\tThe error occurred when using $matrice_path/$file as the input matrices and $golden_path/$golden_file as the expected matrix\n";
                }
            }
        }
    }
}

#
# Compare two matrices
#
# \param actual matrix
# \param expected (golden) matrix
# \param number of rows. 
# \param number of colums. 
#
# \return miss index (x, y)
#
sub compare_matrices {
    my $A = shift;
    my $B = shift;
    my @dims = @_;
 
    for(my $r = 0; $r < $dims[0]; $r++) {
        for(my $c = 0; $c < $dims[1]; $c++) {
            if($A->[$r * $c + $c] ne $A->[$r * $c + $c]) {
                return [$r, $c];
            }
        }
    }

    return undef;
}


# 
# Takes a string and parses it into an array of numeric values.
# assumes that $/ is the delimiter. 
#
# \param string to parse. 
#
# \return the array of values. 
sub string_for_vector {
    my $string = shift; 
    
    my $values = [map {$_} split $/, trim($string)];

    return $values;
}

# Removes blank spaces from edges of a string. 
sub trim { 
    my $totrim = shift;

    $totrim =~ s/^\s+//g;
    $totrim =~ s/\s+$//g;

    return $totrim;
}

#
# Read a file as a vector of values. Assumes that $/ is the delimiter. 
#
# \param path to the file
#
# \return a list or undef. 
sub file_for_vector {
    my $filepath = shift;

    open( my $file, '<', $filepath ) or warn "Failed to open $filepath\n";
    
    my @vector;
    while( my $num = <$file> ) {
        chomp $num;
        if($num =~ /[0-9]+[.]?[0-9]*/) {
            push @vector, $num * 1.0;
        }
    }

    close $file;

    return \@vector;
}

__END__

=head1 NAME

    Runs a series of tests on the specified matrix muliplication bin. 

=head1 SYNOPSIS
    
    ./run_tests.pl 

EXAMPLE:

    ./run_tests.pl -b ../matrixMultiply -m ./matrices -g ./golden -d \  

=head1 OPTIONS
    
=item -h --help

    Print usage information to the screen. 

=item -b --bin-path

    Path to the binary to execute. 

=item -m --matrices-path

    Path to the directory that contains the matrices to multiply. 

=item -g --golden-path

    Path to the directory that contains the expected results.

=item -d --delimiter
    
    The delimiter used to seperate values. 


=item -c --cpu

    Test using the CPU.

=back

=cut

