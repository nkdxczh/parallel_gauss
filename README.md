# parallel_gauss

Description
Using tbb for parallelization.

Version 1
row based; serial max; map(parallel_for) elimination; serial substitution

Version 2
row based; scan max; map(parallel_for) elimination; serial substitution

Version 3
row based; reduce max; map(parallel_for) elimination; serial substitution

Version 4
row based; scan max; map(parallel_for) elimination; map(parallel_for) substitution

Version 5
column based; scan max; map(parallel_for) elimination; serial substitution

Version 6
column based; reduce max; map(parallel_for) elimination; serial substitution

Version 7
column based; serial max; map(parallel_for) elimination; serial substitution

Version 8
row based; scan max; serial elimination; serial substitution

Version 9
row based; reduce max; serial elimination; serial substitution

Version 10
row based; serial max; serial elimination; map(parallel_for) substitution
