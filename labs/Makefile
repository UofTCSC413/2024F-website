all: lab01 lab02 lab03 lab05 lab07 lab09 lab10

lab01: lab01/lab01-soln.ipynb lab01/lab01.ipynb
lab01/lab01-soln.ipynb: lab01/lab01.md
	pushd lab01; notedown lab01.md > lab01-soln.ipynb; popd;
lab01/lab01.ipynb: lab01/lab01.md
	pushd lab01; grep -v "# SOL" lab01.md > lab01_.md; notedown lab01_.md > lab01.ipynb; rm lab01_.md; popd;


lab02: lab02/lab02-soln.ipynb lab02/lab02.ipynb
lab02/lab02-soln.ipynb: lab02/lab02.md
	pushd lab02; notedown lab02.md > lab02-soln.ipynb; popd;
lab02/lab02.ipynb: lab02/lab02.md
	pushd lab02; grep -v "# SOL" lab02.md > lab02_.md; notedown lab02_.md > lab02.ipynb; rm lab02_.md; popd;


lab03: lab03/lab03-soln.ipynb lab03/lab03.ipynb
lab03/lab03-soln.ipynb: lab03/lab03.md
	pushd lab03; notedown lab03.md > lab03-soln.ipynb; popd;
lab03/lab03.ipynb: lab03/lab03.md
	pushd lab03; grep -v "# SOL" lab03.md > lab03_.md; notedown lab03_.md > lab03.ipynb; rm lab03_.md; popd;


lab05: lab05_dp/lab05-soln.ipynb lab05_dp/lab05.ipynb
lab05_dp/lab05-soln.ipynb: lab05_dp/lab05.md 
	pushd lab05_dp; notedown lab05.md > lab05-soln.ipynb; popd;
lab05_dp/lab05.ipynb: lab05_dp/lab05.md
	pushd lab05_dp; grep -v "# SOL" lab05.md > lab05_.md; notedown lab05_.md > lab05.ipynb; rm lab05_.md; popd;


lab07: lab07/lab07-soln.ipynb lab07/lab07.ipynb
lab07/lab07-soln.ipynb: lab07/lab07.md
	pushd lab07; notedown lab07.md > lab07-soln.ipynb; popd;
lab07/lab07.ipynb: lab07/lab07.md
	pushd lab07; grep -v "# SOL" lab07.md > lab07_.md; notedown lab07_.md > lab07.ipynb; rm lab07_.md; popd;


lab09: lab09/lab09-soln.ipynb lab09/lab09.ipynb
lab09/lab09-soln.ipynb: lab09/lab09.md
	pushd lab09; notedown lab09.md > lab09-soln.ipynb; popd;
lab09/lab09.ipynb: lab09/lab09.md
	pushd lab09; grep -v "# SOL" lab09.md > lab09_.md; notedown lab09_.md > lab09.ipynb; rm lab09_.md; popd;



lab10: lab10/lab10-soln.ipynb lab10/lab10.ipynb
lab10/lab10-soln.ipynb: lab10/lab10.md
	pushd lab10; notedown lab10.md > lab10-soln.ipynb; popd;
lab10/lab10.ipynb: lab10/lab10.md
	pushd lab10; grep -v "# SOL" lab10.md > lab10_.md; notedown lab10_.md > lab10.ipynb; rm lab10_.md; popd;



