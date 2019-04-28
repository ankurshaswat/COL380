#include "lab4_mpi.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>	/* ceil */
#include <algorithm> // std::max
#include <mpi.h>
using namespace std;
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
	// if (my_rank == 0)
	// {
	// if (my_rank == 0)
	// 	printf("this2\n");
	MPI_Allgather(MPI_IN_PLACE, end - start, MPI_INT, witness_array, end - start, MPI_INT, MPI_COMM_WORLD);
	// if (my_rank == 0)
	// 	printf("this2\n");
	// }
	// else
	// {
	// 	MPI_Allgather(&witness_array[start], end - start, MPI_INT, witness_array, end - start, MPI_INT, MPI_COMM_WORLD);
	// }
}

void np_textanalysis(char *T, int n, char *P, int m, int *witness_array, int ceil_m_by_2, int *MATCH)
{
	int num_blocks = ceil((1.0 * n) / ceil_m_by_2);
	int bi, i, j, total_len_till_here;

	// int* potential_positions = (int *)malloc(num_blocks + size);

	// if (potential_positions_size < num_blocks + size)
	// {
	// 	potential_positions = (int *)realloc(potential_positions, 2 * sizeof(int) * (num_blocks + size));
	// 	potential_positions_size = 2 * (num_blocks + size);
	// }

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
		// 	potential_positions[bi] = i;
		// }
		// for (bi = start; bi < end && bi < num_blocks; bi++)
		// {
		// 	i = potential_positions[bi];

		if (string_match(T, i, n, P, 0, m))
		{
			MATCH[i] = 1;
		}
	}

	// {
	MPI_Allgather(MPI_IN_PLACE, (end - start) * ceil_m_by_2, MPI_INT, MATCH, (end - start) * ceil_m_by_2, MPI_INT, MPI_COMM_WORLD);
	// free(potential_positions);
	// }
	// else
	// {
	// 	MPI_Allgather(&MATCH[start * ceil_m_by_2], (end - start) * ceil_m_by_2, MPI_INT, MATCH, (end - start) * ceil_m_by_2, MPI_INT, MPI_COMM_WORLD);
	// }
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
	int *S = (int *)malloc(4), S_size = 0;
	int *C = (int *)malloc(4), C_size = 0;
	int *M = (int *)malloc(4), M_size = 0;
	int *sub_MATCH = (int *)malloc(4), sub_MATCH_size = 0;
	int *MATCH = (int *)malloc(4), MATCH_size = 0;
	int *witness_array;
	char *P;

	int *gather_counts = (int *)malloc(sizeof(int) * size);
	int *disps = (int *)malloc(sizeof(int) * size);

	potential_positions = (int *)malloc(4);
	*match_counts = (int *)malloc(sizeof(int) * num_patterns);
	(*matches) = (int *)malloc(sizeof(int) * 10);
	int matches_size = 10;
	int saved = 0;
	int saved_till_now = 0;
	for (pattern_num = 0; pattern_num < num_patterns; pattern_num++)
	{

		m = m_set[pattern_num];
		p = p_set[pattern_num];
		P = pattern_set[pattern_num];

		ceil_m_by_2 = ceil(m / 2.0);

		if (ceil_m_by_2 <= p)
		{
			/* Pattern non periodic */
			// printf("Non periodic \n");
			witness_array = (int *)malloc(sizeof(int) * (ceil_m_by_2 + size));
			MATCH = (int *)malloc(sizeof(int) * 2 * (n + ceil_m_by_2 + size));
			// if (witness_array_size < ceil_m_by_2 + size)
			// {
			// 	witness_array = (int *)realloc(witness_array, 2 * sizeof(int) * (ceil_m_by_2 + size));
			// 	witness_array_size = 2 * (ceil_m_by_2 + size);
			// }
			// if (MATCH_size < n + ceil_m_by_2 + size)
			// {
			// 	MATCH = (int *)realloc(MATCH, 2 * sizeof(int) * (n + ceil_m_by_2 + size));
			// 	MATCH_size = 2 * (n + ceil_m_by_2 + size);
			// }

			get_witness_array(witness_array, ceil_m_by_2, P);
			np_textanalysis(T, n, P, m, witness_array, ceil_m_by_2, MATCH);

			if (my_rank == 0)
			{
				int loc_count = 0;
				for (int i = 0; i < n; i++)
				{
					if (MATCH[i])
					{
						// printf("%d\n", i);
						loc_count++;
					}
				}

				(*match_counts)[pattern_num] = loc_count;

				if (matches_size < saved_till_now + loc_count)
				{
					matches_size = max(2 * matches_size, matches_size + loc_count);
					(*matches) = (int *)realloc(*matches, sizeof(int) * matches_size);
				}

				for (int i = 0; i < n; i++)
				{
					if (MATCH[i])
					{
						(*matches)[saved_till_now++] = i;
					}
				}
			}

			free(witness_array);
			free(MATCH);
		}
		else
		{
			/* Pattern periodic */
			pp_len = 2 * p - 1;
			half_pp_len = ceil(pp_len / 2.0);
			k = floor((1.0 * m) / p);
			k2 = ceil((1.0 * n) / p);

			witness_array = (int *)malloc(sizeof(int) * (half_pp_len + size));
			sub_MATCH = (int *)malloc(sizeof(int) * (n + size * half_pp_len + size));
			C = (int *)malloc(sizeof(int) * (half_pp_len * k2 + size + k2 * size));
			MATCH = (int *)malloc(sizeof(int) * (2 * (n + size)));
			M = (int *)malloc(sizeof(int) * 2 * (n + size));
			S = (int *)malloc(sizeof(int) * 2 * (k2 + size));

			// if (my_rank == 0)
			// 	printf("This start\n");

			get_witness_array(witness_array, half_pp_len, P);

			// if (my_rank == 0)
			// 	printf("This end\n");

			np_textanalysis(T, n, P, pp_len, witness_array, half_pp_len, sub_MATCH);
			get_my_chunk(0, n, &start, &end);

			for (int i = start; i < end && i < n; i++)
			{
				M[i] = 0;
				if (sub_MATCH[i] && string_match(T, i, n, P, (k - 2) * p, m))
				{
					M[i] = 1;
				}
			}

			// if (my_rank == 0)
			// 	printf("This done\n");

			MPI_Allgather(MPI_IN_PLACE, end - start, MPI_INT, M, end - start, MPI_INT, MPI_COMM_WORLD);

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

			// if (my_rank == 0)
			// {
			MPI_Allgather(MPI_IN_PLACE, (end - start) * k2, MPI_INT, C, (end - start) * k2, MPI_INT, MPI_COMM_WORLD);
			// }
			// else
			// {
			// 	MPI_Allgather(&C[start * k2], (end - start) * k2, MPI_INT, C, (end - start) * k2, MPI_INT, MPI_COMM_WORLD);
			// }

			get_my_chunk(0, n - m + 1, &start, &end);
			int count = 0;
			for (int j = start; j < end && j < n - m + 1; j++)
			{
				MATCH[j] = 0;

				int l = j / p;
				int i = j % p;
				if (C[INDEX(i, l, half_pp_len, k2)])
				{
					MATCH[count++] = j;
				}
			}

			if (my_rank == 0)
			{
				MPI_Gather(&count, 1, MPI_INT, gather_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
			}
			else
			{
				MPI_Gather(&count, 1, MPI_INT, gather_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
			}
			int totatl_count = 0;

			if (my_rank == 0)
			{
				// printf("%d\n", size);
				// printf("%d\n", pattern_num);
				// for (int i = 0; i < size; i++)
				// {
				// 	printf("%d ", gather_counts[i]);
				// }
				// printf("\n");

				totatl_count += gather_counts[0];
				disps[0] = 0;

				for (int i = 1; i < size; i++)
				{
					disps[i] = disps[i - 1] + gather_counts[i - 1];
					totatl_count += gather_counts[i];
				}

				// printf("tot count %d\n", totatl_count);
				(*match_counts)[pattern_num] = totatl_count;

				// for (int i = 0; i < size; i++)
				// {
				// 	printf("%d ", disps[i]);
				// }
				// printf("\n");

				if (matches_size < saved_till_now + totatl_count)
				{
					matches_size = max(2 * matches_size, matches_size + totatl_count);
					(*matches) = (int *)realloc(*matches, sizeof(int) * matches_size);
					// if ((*matches) == NULL)
					// {
					// 	printf("Fail\n");
					// }
				}
			}

			// MPI_Allgather(&MATCH[start], end - start, MPI_INT, MATCH, end - start, MPI_INT, MPI_COMM_WORLD);
			if (my_rank == 0)
			{
				MPI_Gatherv(MATCH, count, MPI_INT, &(*matches)[saved_till_now], gather_counts, disps, MPI_INT, 0, MPI_COMM_WORLD);
			}
			else
			{
				MPI_Gatherv(MATCH, count, MPI_INT, MATCH, gather_counts, disps, MPI_INT, 0, MPI_COMM_WORLD);
			}

			if (my_rank == 0)
			{
				saved_till_now += totatl_count;
				// for (int i = 0; i < totatl_count; i++)
				// {
				// 	(*matches)[i] = MATCH[i];
				// }
				printf("%d %d end\n", saved_till_now, matches_size);
			}

			free(witness_array);
			free(sub_MATCH);
			free(C);
			free(MATCH);
			free(M);
			free(S);
			// free(witness_array);
			// free(sub_MATCH);
		}

		// if (my_rank == 0)
		// {
		// 	int printed = 0;
		// 	int count = 0;
		// 	// printf("%d\n", pattern_num);
		// 	for (int i = 0; i < n - m + 1; i++)
		// 	{
		// 		if (MATCH[i])
		// 		{
		// 			(*matches)[saved++] = i;

		// 			if (matches_size == saved)
		// 			{
		// 				printf("%d\n", matches_size);
		// 				(*matches) = (int *)realloc((*matches), sizeof(int) * matches_size * 2);
		// 				matches_size = 2 * matches_size;
		// 			}

		// 			count++;
		// 		}
		// 	}
		// 	(*match_counts)[pattern_num] = count;
		// }
	}

	// if (my_rank == 0)
	// {
	// 	int offset = 0;
	// 	for (int i = 0; i < num_patterns; i++)
	// 	{
	// 		int num = (*match_counts)[i];
	// 		printf("count %d\n", num);

	// 		for (int j = 0; j < num; j++)
	// 		{
	// 			printf("%d ", (*matches)[offset + j]);
	// 		}
	// 		offset += num;
	// 		printf("\n");
	// 	}
	// }

	// free(S);
	// free(C);
	// free(M);
	// // free(sub_MATCH);
	// free(MATCH);
	// // free(witness_array);
	// free(potential_positions);
}
