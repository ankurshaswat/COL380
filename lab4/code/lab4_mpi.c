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
}

void printDeb(int *arr, int n)
{
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		if (arr[i] != 0)
		{
			// printf("%d ", arr[i]);
			printf("%d\n", i);
		}
	}
	printf("\n");
}

void get_witness_array(int *witness_array, int witness_array_len, char *pattern)
{
	int i, k;

	int start, end;
	get_my_chunk(0, witness_array_len, &start, &end);

	for (i = start; i < end && i < witness_array_len; i++)
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
				break;
			}
		}
	}
	MPI_Allgather(&witness_array[start], end - start, MPI_INT, witness_array, end - start, MPI_INT, MPI_COMM_WORLD);
}

void np_textanalysis(char *T, int n, char *P, int m, int *witness_array, int ceil_m_by_2, int *MATCH)
{
	int num_blocks = ceil((1.0 * n) / ceil_m_by_2);
	int bi, i, j, total_len_till_here;

	if (potential_positions_size < num_blocks + size)
	{
		potential_positions = realloc(potential_positions, sizeof(int) * (num_blocks + size));
		potential_positions_size = num_blocks + size;
	}

	int start, end;
	get_my_chunk(0, num_blocks, &start, &end);

	for (i = start * ceil_m_by_2; i < end * ceil_m_by_2 && i < n; i++)
	{
		MATCH[i] = 0;
	}
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
	for (bi = start; bi < end && bi < num_blocks; bi++)
	{
		i = potential_positions[bi];

		if (string_match(T, i, n, P, 0, m))
		{
			MATCH[i] = 1;
		}
	}
	MPI_Allgather(&MATCH[start * ceil_m_by_2], (end - start) * ceil_m_by_2, MPI_INT, MATCH, (end - start) * ceil_m_by_2, MPI_INT, MPI_COMM_WORLD);
}

void periodic_pattern_matching(int n, char *T, int num_patterns, int *m_set,
							   int *p_set, char **pattern_set,
							   int **match_counts, int **matches)
{
	int pattern_num, m, p, ceil_m_by_2, pp_len, k2, half_pp_len, k;
	/*DETERMINE RANK OF THIS PROCESSOR*/
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/*DETERMINE TOTAL NUMBER OF PROCESSORS*/
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int start, end;
	int *S = malloc(4), S_size = 0;
	int *C = malloc(4), C_size = 0;
	int *M = malloc(4), M_size = 0;
	int *sub_MATCH = malloc(4), sub_MATCH_size = 0;
	int *MATCH = malloc(4), MATCH_size = 0;
	int *witness_array = malloc(4), witness_array_size = 0;
	char *P;

	potential_positions = malloc(4);
	*match_counts = malloc(sizeof(int) * num_patterns);
	*matches = malloc(sizeof(int) * 10);
	int matches_size = 10;
	int saved = 0;

	for (pattern_num = 0; pattern_num < num_patterns; pattern_num++)
	{

		m = m_set[pattern_num];
		p = p_set[pattern_num];
		P = pattern_set[pattern_num];

		ceil_m_by_2 = ceil(m / 2.0);

		if (ceil_m_by_2 <= p)
		{
			/* Pattern non periodic */
			if (witness_array_size < ceil_m_by_2 + size)
			{
				witness_array = realloc(witness_array, sizeof(int) * (ceil_m_by_2 + size));
				witness_array_size = ceil_m_by_2 + size;
			}
			if (MATCH_size < n + ceil_m_by_2 + size)
			{
				MATCH = realloc(MATCH, sizeof(int) * (n + ceil_m_by_2 + size));
				MATCH_size = n + ceil_m_by_2 + size;
			}

			get_witness_array(witness_array, ceil_m_by_2, P);
			np_textanalysis(T, n, P, m, witness_array, ceil_m_by_2, MATCH);
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
				witness_array = realloc(witness_array, sizeof(int) * (half_pp_len + size));
				witness_array_size = half_pp_len + size;
			}

			get_witness_array(witness_array, half_pp_len, P);

			if (sub_MATCH_size < n + size * half_pp_len + size)
			{
				sub_MATCH = realloc(sub_MATCH, sizeof(int) * (n + size * half_pp_len + size));
				sub_MATCH_size = n + size * half_pp_len + size;
			}

			if (MATCH_size < n + size)
			{
				MATCH = realloc(MATCH, sizeof(int) * (n + size));
				MATCH_size = n + size;
			}

			np_textanalysis(T, n, P, pp_len, witness_array, half_pp_len, sub_MATCH);

			if (M_size < n + size)
			{
				M = realloc(M, sizeof(int) * (n + size));
				M_size = n + size;
			}

			get_my_chunk(0, n, &start, &end);

			for (int i = start; i < end && i < n; i++)
			{
				M[i] = 0;
				if (sub_MATCH[i] && string_match(T, i, n, P, (k - 2) * p, m))
				{
					M[i] = 1;
				}
			}

			MPI_Allgather(&M[start], end - start, MPI_INT, M, end - start, MPI_INT, MPI_COMM_WORLD);

			if (C_size < half_pp_len * k2 + size + k2 * size)
			{
				C = realloc(C, sizeof(int) * (half_pp_len * k2 + size + k2 * size));
				C_size = half_pp_len * k2 + size + k2 * size;
			}

			if (S_size < k2 + size)
			{
				S = realloc(S, sizeof(int) * (k2 + size));
				S_size = k2 + size;
			}

			get_my_chunk(0, p, &start, &end);

			for (int i = start; i < end && i < p; i++)
			{
				int len = 0;
				int num_consecutive_ones = 0;
				for (int x = k2 - 1; x >= 0; x--)
				{
					if (i + x * p < n)
					{
						len++;
						S[x] = M[i + x * p];
						if (S[x])
						{
							num_consecutive_ones++;
						}
						else
						{
							num_consecutive_ones = 0;
						}

						C[INDEX(i, x, half_pp_len, k2)] = (num_consecutive_ones >= k - 1);
					}
				}
			}
			MPI_Allgather(&C[start * k2], (end - start) * k2, MPI_INT, C, (end - start) * k2, MPI_INT, MPI_COMM_WORLD);

			get_my_chunk(0, n - m + 1, &start, &end);
			for (int j = start; j < end && j < n - m + 1; j++)
			{
				MATCH[j] = 0;

				int l = j / p;
				int i = j % p;
				MATCH[j] = C[INDEX(i, l, half_pp_len, k2)];
			}
			// MPI_Allgather(&MATCH[start], end - start, MPI_INT, MATCH, end - start, MPI_INT, MPI_COMM_WORLD);
			MPI_Gather(&MATCH[start], end - start, MPI_INT, MATCH, end - start, MPI_INT, 0, MPI_COMM_WORLD);
		}

		if (my_rank == 0)
		{
			int printed = 0;
			int count = 0;
			for (int i = 0; i < n - m + 1; i++)
			{
				if (MATCH[i])
				{
					// 	printed = 1;
					printf("%d ", i);

					(*matches)[saved] = i;
					if (matches_size == saved)
					{
						*matches = realloc(*matches, 2 * matches_size);
						matches_size *= 2;
					}
					count++;
				}
			}
			// (*match_counts)[pattern_num] = count;
			(*match_counts)[pattern_num] = count;
			printf("Count %d\n", count);
			// if (printed)
			// {
			// 	printf("\n:%d for iter %d \n", count, pattern_num);
			// }
		}
	}

	free(S);
	free(C);
	free(M);
	free(sub_MATCH);
	free(MATCH);
	free(witness_array);
	free(potential_positions);
}
