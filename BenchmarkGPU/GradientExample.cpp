#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;

#define MAX_TE 50 // maximum training examples
#define MAX_W 50 // maximum weights

// gradient descent for the expression w0 + w1x1 + w2x2 + w3x3 + w4x4 + ... + wnxn = ai

double a = 0.005; // learning rate 0.005 would converge to a global minima; however it takes more iterations
				  // You could also set the variable a to 0.0005 * (sum of ai) / i where (sum of ai) / i is actually the arithmetic
				  //of all examples. Then do a check whether a exceeds 1.0. If it exceeds 1.0 then it would be best to set 1.0, because //we will have pretty large values for the functions and we might not find a global minima enough fast(with less //iterations)

double examples[MAX_TE][MAX_W]; // training examples and x values for all of them
double ans[MAX_TE]; // answers of the functions
double weights[MAX_W];
int e, w;

double eval(int i)
{
	// evaluates the output with the current weights for example i
	double current = weights[0];
	for (int j = 1; j < w; ++j)
	{
		current += weights[j] * examples[i][j];
	}
	return current;
}

int run()
{
	cin >> e >> w;
	for (int i = 0; i < e; ++i)
	{
		for (int j = 1; j < w; ++j)
		{
			cin >> examples[i][j];
		}
		cin >> ans[i];
	}
	for (int i = 0; i < w; ++i)
	{
		// sets all weights to 0.0; we could set them any small random value: 0.001 0.1 0.02 and so on
		weights[i] = 0.0;
	}
	for (int p = 0; p < 500000; ++p)
	{
		// we will run the algorithm 500000 times in order to be sure that we have found the perfect weights.
		// you could add some more code to find convergence
		for (int i = 0; i < e; ++i)
		{
			double y = eval(i);
			// compute w0
			weights[0] = weights[0] + a*(ans[i] - y);
			for (int j = 1; j < w; ++j)
			{
				// compute all weights from 1 to w
				weights[j] = weights[j] + a * (ans[i] - y) * examples[i][j];
			}
		}
	}
	for (int i = 0; i < w; ++i)
	{
		cout << "w" << i << " = " << weights[i] << " ";
	}
	return 0;
}

/*
EXAMPLE INPUT:
4 2
-1 0
0 1
1 2
2 1

OUTPUT:
w0 = 0.8 w1 = 0.4
*/