#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>

using namespace std;

typedef double(*TestFunctTempl)(long&);

double PIsuccessively(long num_steps) {
	double t_start = omp_get_wtime();

	long i;
	double step, pi, x, sum = 0.0;;
	step = 1.0 / (double)num_steps;

	for (i = 0; i < num_steps; i++)
	{
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	double t_end = omp_get_wtime();
	return t_end - t_start;
}
double PIparallelStatic(long num_steps) {
	double t_start = omp_get_wtime();

	long i;
	double step, pi, x, sum = 0.0;
	step = 1.0 / (double)num_steps;
#pragma omp parallel for schedule(static, num_steps/20) private(x) reduction(+:sum)
	for (i = 0; i < num_steps; i++)
	{
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	double t_end = omp_get_wtime();
	return t_end - t_start;
}
double PIparallelDynamic(long num_steps) {
	double t_start = omp_get_wtime();

	long i;
	double step, pi, x, sum = 0.0;
	step = 1.0 / (double)num_steps;
#pragma omp parallel for schedule(dynamic, num_steps/20) private(x) reduction(+:sum)
	for (i = 0; i < num_steps; i++)
	{
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	double t_end = omp_get_wtime();
	return t_end - t_start;
}
double PIparallelGuided(long num_steps) {
	double t_start = omp_get_wtime();

	long i;
	double step, pi, x, sum = 0.0;
	step = 1.0 / (double)num_steps;
	#pragma omp parallel for schedule(guided, num_steps/20) private(x) reduction(+:sum)
	for (i = 0; i < num_steps; i++)
	{
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	double t_end = omp_get_wtime();
	return t_end - t_start;
}
double PIparallelSections(long num_steps) {
	double t_start = omp_get_wtime();

	double step, pi, x, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
	int n_t = 4;
	step = 1.0 / (double)num_steps;
	#pragma omp parallel 
	{
		n_t = omp_get_max_threads();
	}
	int st1 = num_steps / n_t;
	int st2 = num_steps * 2 / n_t;
	int st3 = num_steps * 3 / n_t;

	#pragma omp parallel sections private(x)
	{
	#pragma omp section
		{
			for (int i = 0; i < st1; i++)
			{
				x = (i + 0.5) * step;
				sum1 += 4.0 / (1.0 + x * x);
			}
		}
	#pragma omp section
		{
			for (int i = st1; i < st2; i++)
			{
				x = (i + 0.5) * step;
				sum2 = 4.0 / (1.0 + x * x);
			}
		}
	#pragma omp section
		{
			if (n_t > 2)
			{
				for (int i = st2; i < st3; i++)
				{
					x = (i + 0.5) * step;
					sum3 = 4.0 / (1.0 + x * x);
				}
			}
		}
	#pragma omp section
		{
			if (n_t > 3)
			{
				for (int i = st3; i < num_steps; i++)
				{
					x = (i + 0.5) * step;
					sum4 = 4.0 / (1.0 + x * x);
				}
			}
		}
	}
	pi = step * (sum1 + sum2 + sum3 + sum4);
	double t_end = omp_get_wtime();
	return t_end - t_start;
}

double TestPIsuccessively(long& num_steps)
{
	return PIsuccessively(num_steps);
}
double TestPIparallelStatic(long& num_steps)
{
	return PIparallelStatic(num_steps);
}

double TestPIparallelDynamic(long& num_steps)
{
	return PIparallelDynamic(num_steps);
}
double TestPIparallelGuided(long& num_steps)
{
	return PIparallelGuided(num_steps);
}
double TestPIparallelSections(long& num_steps)
{
	return PIparallelSections(num_steps);
}

double AvgTrustedInterval(double& avg, vector<double>& times, int& cnt)
{
	double sd = 0, newAVg = 0;
	int newCnt = 0;
	for (int i = 0; i < cnt; i++)
	{
		sd += (times[i] - avg) * (times[i] - avg);
	}
	sd /= (cnt - 1.0);
	sd = sqrt(sd);
	for (int i = 0; i < cnt; i++)
	{
		if (avg - sd <= times[i] && times[i] <= avg + sd)
		{
			newAVg += times[i];
			newCnt++;
		}
	}
	if (newCnt == 0) newCnt = 1;
	return newAVg / newCnt;
}

double TestIter(void* Funct, long num_steps)
{
	double curtime = 0, avgTime = 0, avgTimeT = 0, correctAVG = 0;
	int iterations = 100;
	vector<double> Times(iterations);

	for (int i = 0; i < iterations; i++)
	{
		// Запуск функции и получение времени в миллисекундах
		curtime = ((*(TestFunctTempl)Funct)(num_steps)) * 1000;
		Times[i] = curtime;
		avgTime += curtime;
		cout << "+";
	}

	cout << endl;
	// Вычисление среднеарифметического по всем итерациям и вывод значения на экран
	avgTime /= iterations;
	cout << "AvgTime:" << avgTime << endl;
	// Определения среднеарифметического значения в доверительном интервале по всем итерациям и вывод значения на экран
	avgTimeT = AvgTrustedInterval(avgTime, Times, iterations);
	cout << "AvgTimeTrusted:" << avgTimeT << endl;
	return avgTimeT;
}

void test_functions(void** Functions, vector<string> fNames)
{
	int nd = 0;
	double times[4][5][3];
	for (int num_steps = 500000; num_steps <= 2000000; num_steps += 500000)
	{
		for (int threads = 1; threads <= 4; threads++)
		{
			omp_set_num_threads(threads);
			//перебор алгоритмов по условиям
			for (int alg = 0; alg <= 4; alg++)
			{
				if (threads == 1)
				{
					if (alg == 0) {
						times[nd][alg][0] = TestIter(Functions[alg], num_steps);
						times[nd][alg][1] = times[nd][alg][0];
						times[nd][alg][2] = times[nd][alg][0];
					}
				}
				else
				{
					if (alg != 0)
					{
						times[nd][alg][threads - 2] = TestIter(Functions[alg], num_steps);
					}
				}
			}
		}
		nd++;
	}

	ofstream fout("output.txt");
	fout.imbue(locale("Russian"));
	for (int ND = 0; ND < 4; ND++)
	{
			switch (ND)
			{
			case 0:
				cout << "\n----------500000 количество итераций----------" << endl;
				break;
			case 1:
				cout << "\n----------1000000 количество итераций----------" << endl;
				break;
			case 2:
				cout << "\n----------1500000 количество итераций----------" << endl;
				break;
			case 3:
				cout << "\n----------2000000 количество итераций----------" << endl;
				break;
			default:
				break;
			}


			for (int alg = 0; alg < 5; alg++)
			{
				for (int threads = 1; threads <= 4; threads++)
				{
					if (threads == 1)
					{
						if (alg == 0) {
							cout << "Поток " << threads << " --------------" << endl;
							cout << fNames[alg] << "\t" << times[ND][alg][0] << " ms." << endl;
							fout << times[ND][alg][0] << endl;

						}
					}
					else
					{
						if (alg != 0)
						{
							cout << "Поток " << threads << " --------------" << endl;
							cout << fNames[alg] << "\t" << times[ND][alg][threads - 2] << " ms." << endl;
							fout << times[ND][alg][threads - 2] << endl;
						}
					}
				}
			}
		}
	fout.close();
}

int main() {

	setlocale(LC_ALL, "RUS");
	cout.imbue(locale("Russian"));
	void** Functions = new void* [5]{ TestPIsuccessively, TestPIparallelStatic, TestPIparallelDynamic, TestPIparallelGuided, TestPIparallelSections};
	vector<string> function_names = { "Последовательная реализация","Параллельная реализация FOR(static)",
		"Параллельная реализация FOR(dynamic)", "Параллельная реализация FOR(guided)", "Параллельная реализация Section" };
	
	test_functions(Functions, function_names);
	return 0;
}

