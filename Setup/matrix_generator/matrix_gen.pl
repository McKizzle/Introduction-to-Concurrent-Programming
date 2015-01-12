#!/usr/bin/env perl

use warnings;
use strict;
use Data::Dumper;
use Getopt::Long;
use Pod::Usage;

my $numArgs = @ARGV;
my $args = {
    'help' => 0,
    'random seed' => 0,
    'matrix sizes' => "",
    'delimiter' => undef
};

GetOptions(
    'help|?|h' => \$args->{'help'},
    'r|seed=i' => \$args->{'random seed'},
    'm|matrix-size=s@' => \$args->{'matrix sizes'},
    'd|delimiter=s' => \$args->{'delimiter'}
    ) or die "Error parsing arguments\n";

if($args->{'help'} || $numArgs == 0 ) {
    pod2usage(1);
} else {
    srand $args->{'random seed'};
    if($args->{'delimiter'}) {
        $args->{'delimiter'} = substr $args->{'delimiter'}, 0, 1;
    } else {
        $args->{'delimiter'} = ' ';
    }

    foreach(@{$args->{'matrix sizes'}}) {
        if($_ =~ /([0-9]+)x([0-9]+),([0-9]+)x([0-9]+)/) {
            my ($A, $B) = generate_two_column_major_matrices($1, $2, $3, $4);
            my $C = matrix_multiply($1, $2, $A, $3, $4, $B);
            dump_matrices_to_file("matrices", "$1x$2-$3x$4.mat", $args->{'delimiter'}, $A, $B);
            dump_matrices_to_file("golden", "$1x$2-$3x$4-$1x$4.mat", $args->{'delimiter'}, $C);
        }
    }
}

#
# Dump the contents of 1 or more matrices to a file.
#
# \param directory to write to. 
# \param file name to use.
# \param the delimiter to use. 
# \param M an array reference of one or more array references. 
#
sub dump_matrices_to_file {
    my $out_dir   = shift;
    my $file_name = shift; 
    my $delimiter = shift;
    my @matrices  = @_;
    
    mkdir $out_dir or warn "Warning: $!"; 

    open my $file, "> $out_dir/$file_name" or die "Failed to open the file $out_dir/$file_name.\n$!.";

    foreach(@matrices) {
        my @M = @$_;
         
        foreach(@M) {
            print $file $_, $delimiter; 
        }
    }
    print $file "\n";
    close $file;
}

#
# Multiply two matrices
#
# \param A rows
# \param A columns 
# \param A data
# \param B rows
# \param B columns
# \parma B data
#
sub matrix_multiply {
    my $ar = shift;
    my $ac = shift;
    my $A  = shift;
    my $br = shift;
    my $bc = shift;
    my $B  = shift;
    
    # Inialize matrix C
    my $cr  = $ar;
    my $cc  = $bc;
    my $len = $cr * $cc;
    my $C = [map { 0.0 } 0...($cr * $cc - 1)];
    
    # Multiply A and B and store into C.
    my $m = $ac;
    my $i = 0;
    my $row = 0;
    my $col = 0;
    my $dot = 0.0;
    do 
    {
        $row = int ($i / $cc);
        $col = $i % $cc;

        $dot = 0.0;
        for(my $j = 0; $j < $m; $j++) {
            $dot += $A->[$row * $ac + $j] * $B->[$bc * $j + $col];
        }
        $C->[$i] = $dot;
    }
    while( ++$i < $len );

    return $C;
}

#
# Generates two matrices given the dimensions that have been passed in. 
# 
# \param A rows
# \param A columns 
# \param B rows
# \param B columns
#
# \return Matrix A and Matrix B (column major) as two linear vectors. 
sub generate_two_column_major_matrices {
    my $ar = shift;
    my $ac = shift;
    my $br = shift;
    my $bc = shift;
    
    my $A = [map { int (rand() * 10) } 0...($ar * $ac - 1)];
    my $B = [map { int (rand() * 10) } 0...($br * $bc - 1)];

    return ($A, $B);
}

__END__

=head1 NAME
    
    Generates matrices and saves them as RxC.vec vector files where `R` is the row count and `C` is the column count. 

=head1 SYNOPSIS

    matrix_gen.pl [-h] [-v N [N...]] [-r R]

EXAMPLE:
    
    ./matrix_gen.pl -m 2x4,4x2 -m 12x100,100x12 -m 1512x1512,1512x1512 -r 23
    ./matrix_gen.pl -m 2x4,4x2 -m 12x100,100x12 -m 23x57,57x9 -m 84x16,16x22 \
        -m 1024x1024,1024x1024 -m 1543x1200,1200x1423 -r 128

=head1 OPTIONS
    
    
=item -h --help
    Print usage information to the screen. 

=item -m --matrix-size N [N...]
    The set of matrix sizes to generate. N is in the format `RxC,R'xC'` where `R` is the number of rows and `C` is the number of columns and `R'`, `C'` are the dimensions of the matrix to multiply it by. 

=item -r R, --random-seed R
    The random seed to use when generating the matrices.

=item -d C, --delimiter C
    A single character representing the delimiter to use to seprate the matrix elements. 

=back

=cut

