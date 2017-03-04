#include <timerlib.h>
void cm_column_decomposition(int n,
			     double a[n+1][n],
			     int m,
			     int pv[],
			     int i);
void cm_column_decomposition_recursive(int n,
				       double a[n+1][n],
				       int m,
				       int pv[],
				       int i);

void cm_process_right_part(int n,
			   double a[n+1][n],
			   int m,
			   int pv[],
			   int i,
			   int iend);
