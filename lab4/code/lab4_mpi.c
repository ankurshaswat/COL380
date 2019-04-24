#include "lab4_mpi.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int *potential_positions;
int potential_positions_size = 0;
int my_rank, size;

static inline int INDEX(int i1, int i2, int l1, int l2) { return i1 * l2 + i2; }

int DUEL(char *Z, int n1, char *Y, int *witness_array, int i, int j)
{
	int k = witness_array[j - i];
	if (j + k >= n1 || Z[j + k] != Y[k])
	{
		return i;
	}
	return j;
}

int string_match(char *target, int start1, int end1, char *pattern, int start2, int end2)
{
	int l1 = end1 - start1;
	int l2 = end2 - start2;

	if (l2 > l1)
		return 0;

	int i;

	for (i = 0; i < end2 - start2; i++)
		if (start1 + i >= end1 || target[start1 + i] != pattern[start2 + i])
			return 0;
	return 1;
}

void get_my_chunk(int start_arr, int end_arr, int *my_start, int *my_end)
{
	int num = end_arr - start_arr;

	int per_proc = ceil((1.0 * num) / size);

	*my_start = my_rank * per_proc + start_arr;
	*my_end = *my_start + per_proc;

	// if (num < size)
	// {
	// 	if (my_rank < num)
	// 	{
	// 		*my_start = start_arr + my_rank;
	// 		*my_end = *my_start + 1;
	// 	}
	// 	else
	// 	{
	// 		*my_start = 0;
	// 		*my_end = 0;
	// 	}
	// }
	// else
	// {
	// 	int excess = num % size;
	// 	int uniform = num / size;
	// 	if (my_rank < excess)
	// 	{
	// 		*my_start = (uniform + 1) * my_rank;
	// 		*my_end = *my_start + uniform + 1;
	// 	}
	// 	else
	// 	{
	// 		*my_start = (uniform + 1) * excess + (my_rank - excess) * uniform;
	// 		*my_end = *my_start + uniform;
	// 	}
	// }
}

void printDeb(int *arr, int n)
{
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		printf("%d ", arr[i]);
	}
	printf("\n");
}

void get_witness_array(int *witness_array, int witness_array_len, char *pattern)
{
	int i, k;

	int start, end;
	get_my_chunk(0, witness_array_len, &start, &end);
	// printf("%d %d\n", start, end);

	// printf("%d\n", witness_array_len);

	for (i = start; i < end && i < witness_array_len; i++)
	// for (i = 1; i < witness_array_len; i++)
	{
		witness_array[i] = 0;
		if (i == 0)
		{
			continue;
		}
		for (k = 0; k < witness_array_len; k++)
		{
			if (pattern[k] != pattern[i + k])
			{
				witness_array[i] = k;
				// printf("witness_array[%d]=%d\n", i, k);
				break;
			}
		}
	}
	// printDeb(witness_array, witness_array_len);
	MPI_Allgather(&witness_array[start], end - start, MPI_INT, witness_array, end - start, MPI_INT, MPI_COMM_WORLD);
	// printDeb(witness_array, witness_array_len);
}

void np_textanalysis(char *T, int n, char *P, int m, int *witness_array, int ceil_m_by_2, int *MATCH)
{
	int num_blocks = ceil((1.0 * n) / ceil_m_by_2);
	int bi, i, j, total_len_till_here;

	if (potential_positions_size < num_blocks + size)
	{
		// printf("1 %ld\n", sizeof(int) * num_blocks);
		potential_positions = realloc(potential_positions, sizeof(int) * (num_blocks + size));
		potential_positions_size = num_blocks + size;
	}

	int start_big, end_big;
	get_my_chunk(0, num_blocks * ceil_m_by_2, &start_big, &end_big);
	for (i = start_big; i < end_big && i < n; i++)
	{
		MATCH[i] = 0;
	}

	int start, end;
	get_my_chunk(0, num_blocks, &start, &end);

	for (bi = start; bi < end && bi < num_blocks; bi++)
	{
		i = bi * ceil_m_by_2;
		total_len_till_here = (bi + 1) * ceil_m_by_2;
		if (bi == num_blocks - 1)
		{
			total_len_till_here = n;
		}
		for (j = i + 1; j < total_len_till_here; j++)
		{
			i = DUEL(T, n, P, witness_array, i, j);
		}
		potential_positions[bi] = i;
	}
	// printf("num_blocks my_rank=%d start=%d end=%d\n", my_rank, start, end);

	// MPI_Allgather(&potential_positions[start], end - start, MPI_INT, potential_positions, end - start, MPI_INT, MPI_COMM_WORLD);

	for (bi = start; bi < end && bi < num_blocks; bi++)
	{
		i = potential_positions[bi];
		if (string_match(T, i, n, P, 0, m))
		{
			MATCH[i] = 1;
		}
	}

	// printf("my_rank=%d start=%d end=%d\n", my_rank, start, end);
	// if (my_rank == 0)
	// {
	// 	printf("Size of MATCH %d\n", Match_size);
	// }
	// printDeb(MATCH)
	MPI_Allgather(&MATCH[start_big], end_big - start_big, MPI_INT, MATCH, end_big - start_big, MPI_INT, MPI_COMM_WORLD);
}

void periodic_pattern_matching(int n, char *T, int num_patterns, int *m_set,
							   int *p_set, char **pattern_set,
							   int **match_counts, int **matches)
{
	int pattern_num, m, p, ceil_m_by_2, pp_len, k2, half_pp_len, k;
	/*START MPI */
	MPI_Init(NULL, NULL);
	/*DETERMINE RANK OF THIS PROCESSOR*/
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/*DETERMINE TOTAL NUMBER OF PROCESSORS*/
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// printf("my_rank=%d size=%d\n", my_rank, size);
	int start, end;
	// get_my_chunk(0, 4, &start, &end);
	// printf("my_rank=%d size=%d start=%d end=%d\n", my_rank, size, start, end);

	int *S = malloc(0), S_size = 0;
	int *C = malloc(0), C_size = 0;
	int *M = malloc(0), M_size = 0;
	int *sub_MATCH = malloc(0), sub_MATCH_size = 0;
	int *MATCH = malloc(0), MATCH_size = 0;
	int *witness_array = malloc(0), witness_array_size = 0;
	char *P;

	potential_positions = malloc(0);
	// *match_counts = malloc(sizeof(int) * num_patterns);

	for (pattern_num = 0; pattern_num < num_patterns; pattern_num++)
	{

		m = m_set[pattern_num];
		p = p_set[pattern_num];
		P = pattern_set[pattern_num];

		ceil_m_by_2 = ceil(m / 2.0);

		// printf("Iter %d\n", pattern_num);

		if (ceil_m_by_2 <= p)
		{
			// printf("Non Periodic Case\n");

			/* Pattern non periodic */
			if (witness_array_size < ceil_m_by_2 + size)
			{
				// printf("2 %ld\n", sizeof(int) * ceil_m_by_2);

				witness_array = realloc(witness_array, sizeof(int) * (ceil_m_by_2 + size));
				witness_array_size = ceil_m_by_2 + size;
			}
			if (MATCH_size < n + size)
			{
				// printf("3 %ld\n", sizeof(int) * n);

				MATCH = realloc(MATCH, sizeof(int) * (n + size));
				MATCH_size = n + size;
			}

			get_witness_array(witness_array, ceil_m_by_2, P);
			np_textanalysis(T, n, P, m, witness_array, ceil_m_by_2, MATCH);

			// printDeb(witness_array, half_pattern_length);
		}
		else
		{
			/* Pattern periodic */
			pp_len = 2 * p - 1;
			half_pp_len = ceil(pp_len / 2.0);
			k = floor((1.0 * m) / p);
			k2 = ceil((1.0 * n) / p);

			if (witness_array_size < half_pp_len + size)
			{
				// printf("4 %ld\n", sizeof(int) * half_pp_len);

				witness_array = realloc(witness_array, sizeof(int) * (half_pp_len + size));
				witness_array_size = half_pp_len + size;
			}

			get_witness_array(witness_array, half_pp_len, P);
			// printDeb(witness_array, half_pp_len);
			if (sub_MATCH_size < n + size)
			{
				// printf("5 %ld\n", sizeof(int) * n);

				sub_MATCH = realloc(sub_MATCH, sizeof(int) * (n + size));
				sub_MATCH_size = n + size;
			}

			if (MATCH_size < n + size)
			{
				// printf("11 %ld\n", sizeof(int) * n);

				MATCH = realloc(MATCH, sizeof(int) * (n + size));
				MATCH_size = n + size;
			}

			for (int i = 0; i < n; i++)
			{
				sub_MATCH[i] = 0;
			}
			np_textanalysis(T, n, P, pp_len, witness_array, half_pp_len, sub_MATCH);

			if (M_size < n + size)
			{
				// printf("6 %ld\n", sizeof(int) * n);

				M = realloc(M, sizeof(int) * (n + size));
				M_size = n + size;
			}

			for (int i = 0; i < n; i++)
			{
				M[i] = 0;
				if (sub_MATCH[i] && string_match(T, i, n, P, (k - 2) * p, m))
				{
					M[i] = 1;
				}
			}

			if (C_size < half_pp_len * k2 + size)
			{
				// printf("7 %ld\n", sizeof(int) * half_pp_len * k2);

				C = realloc(C, sizeof(int) * (half_pp_len * k2 + size));
				C_size = half_pp_len * k2 + size;
			}

			if (S_size < k2 + size)
			{
				// printf("8 %ld\n", sizeof(int) * k2);

				S = realloc(S, sizeof(int) * (k2 + size));
				S_size = k2 + size;
			}

			for (int i = 0; i < p; i++)
			{
				// C[INDEX(i] = (int *)malloc(sizeof(int) * k2);
				int len = 0;
				for (int x = 0; x < k2; x++)
				{
					if (i + x * p < n)
					{
						len++;
						S[x] = M[i + x * p];
					}
				}
				for (int j = 0; j < len; j++)
				{
					C[INDEX(i, j, half_pp_len, k2)] = 0;
					int found = 1;
					for (int x = 0; x < k - 1; x++)
					{
						if (S[x + j] == 0)
						{
							found = 0;
							break;
						}
					}
					if (found)
					{
						C[INDEX(i, j, half_pp_len, k2)] = 1;
					}
				}
			}
			for (int j = 0; j < n - m + 1; j++)
			{
				MATCH[j] = 0;

				int found = 0;
				int l = -1;
				int i;
				for (i = 0; i < p; i++)
				{
					if ((j - i) % p == 0)
					{
						found = 1;
						l = (j - i) / p;
						break;
					}
				}
				if (found)
				{
					MATCH[j] = C[INDEX(i, l, half_pp_len, k2)];
				}
			}
		}

		if (my_rank == 0)
		{
			int printed = 0;
			for (int i = 0; i < n - m + 1; i++)
			{
				if (MATCH[i])
				{
					printed = 1;
					printf("%d ", i);
				}
			}
			if (printed)
			{
				printf(": for iter %d \n", pattern_num);
			}
		}

		// printf("Iter %d done\n", pattern_num);

		// printDeb(MATCH, n - m + 1);
	}

	MPI_Finalize(); /* EXIT MPI */
}
