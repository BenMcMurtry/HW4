//  Created by Ben McMurtry on 7/12/2016.


/*
 This is a program to calculate the temperature around a nuclear waste rod using the method outlined in example 5.3.1 (pages 41–44) of: Numerical Solutions of Partial Differential Equations (2011) by Louise Olsen-Kettle, University of Queensland
 http://espace.library.uq.edu.au/eserv/UQ:239427/Lectures_Book.pdf
 
 This program was made with the intention of being able to reproduce figure 5.3(b) in the booklet, which shows the Temperature along the radius of a nuclear waste rod and its surroundings at different points in time.
 
 The program makes use of gsl vectors and matrices, for the storage and manipulation of the Temperatures, radial distances, source terms, and boundary conditions needed to solve the Partial Differential equation found on page 43 of the booklet.
 
 The solution of the PDE is achieved using the Backwards Euler method with Finite Differences. This is then solved with the use of a gsl linear algebra function which solves the matrix equation Ax = b using Cholesky decomposition.
*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <errno.h>

/* Includes for GSl. */
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

/* Definitions for printing file and PlotData function. */
#define GNUPLOT_EXE    "/opt/local/bin/gnuplot" /* Please put your own computer filepath for gnuplot here! */
#define GNUPLOT_SCRIPT "HW4_gnuplot.script"
#define GNUPLOT_DATA   "HW4_data.txt"

/* Definitions for the scenario to be solved. */
#define RADIUS_MIN (0.0)        /* [cm] */
#define RADIUS_MAX (100.0)      /* [cm] */
#define RADIUS_ROD (25.0)       /* [cm] */

#define T_ROD (1.0)             /* Initial Temperature change due to nuclear rod [K] */
#define T_ENVIRONMENT (300.0)   /* Environment Temperature [K] */

#define TIME_MIN (0.0)          /* [Years] */
#define TIME_MAX (100.0)        /* [Years] */
#define TAU_0 (100.0)           /* Lifetime [Years] */

#define KAPPA (2.0e7)           /* Thermal Conductivity constant [cm^2/Year] */



/* Initialises all elements in a temperature vector to be the temperature of the environment. */
static void init_rod_temp_env(gsl_vector *T) {
    gsl_vector_set_all(T, T_ENVIRONMENT);
}

/* Creates vectors for the Diagonal, Above-Diagonal and Below-Diagonal of matrix A, and adds a Neumann Boundary condition (NBC). */
static void make_tridiag_matrix(gsl_vector *diag, gsl_vector *above_diag, gsl_vector *below_diag, int n, double s) {
    for (int j = 0; j < n; j++) {
        gsl_vector_set(diag, j, 1.0 + 2.0 * s);
        gsl_vector_set(diag, 0, 1.0 + s + (s / 2.0));   /* NBC at r=0: dT(0,t)/dr = 0 is approximated by T_0^k = T_1^k */
    }
    for (int j = 1; j < n; j++) {
        gsl_vector_set(above_diag, j - 1, -s - (s / (2.0 * j)));
    }
    for (int j = 2; j < n + 1; j++) {
        gsl_vector_set(below_diag, j - 2, -s + (s / (2.0 * j)));
    }
}

/* Sets the source vector for time k, for each element dr along the rod_radius, leaving the source terms outside the rod as 0. */
static void set_source_vector(gsl_vector *source, int j_bound, double dt, int k) {
    gsl_vector_set_zero(source);
    double source_term = ((T_ROD * KAPPA * dt) / pow(RADIUS_ROD, 2)) * exp(-((double)k * dt) / TAU_0);
    for (int j = 0; j < j_bound; j++) {
        gsl_vector_set(source, j, source_term);
    }
}

/* Creates the b vector, by summing the source vector and the previous Temperature vector, and setting the Dirichlet boundary condition. */
static void set_b_vector(gsl_vector *b, gsl_vector *T_prev, gsl_vector *source, int n, double s) {
    gsl_vector_set_zero(b);
    gsl_vector_add(b, source);
    gsl_vector_add(b, T_prev);
    gsl_vector_set(b, n - 1, -(-s - (s / (2.0 * n))) * T_ENVIRONMENT);  /* Dirichlet boundary condition at r=rc: b[n] = (-s-(s/(2*n)))*300. */
}

/* Prints gsl matrix Temp, in the format one would write it, to both stdout and GNUPLOT_DATA. */
static void matrix_print(int rows, int cols, gsl_matrix *temp) {
    FILE *outfile = fopen(GNUPLOT_DATA, "w");
    if (!outfile) {
        fprintf(stderr, "Error: Could not open file '%s'.\n", GNUPLOT_DATA);
        exit(1);
    }
    
    fprintf(stdout, "%-12s %-12s %-12s %-12s %-12s",
            "r [cm]",  "Temp1year", "Temp10year", "Temp50year", "Temp100year\n");
    fprintf(outfile, "%-12s %-12s %-12s %-12s %-12s",
            "r [cm]",  "Temp1year", "Temp10year", "Temp50year", "Temp100year\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%-12.7g ", gsl_matrix_get(temp, i, j));
            fprintf(outfile, "%-12.7g ", gsl_matrix_get(temp, i, j));
        }
        printf("\n");
        fprintf(outfile, "\n");
    }
    printf("\n");
    
    fclose(outfile);
}

/* This function calls gnuplot and a script to plot the contents of GNUPLOT_DATA. */
static void plot_data() {
    char command[PATH_MAX];
    snprintf(command, sizeof(command), "%s %s", GNUPLOT_EXE, GNUPLOT_SCRIPT );
    system( command );
}


int main () {
    printf("Welcome to the nuclear rod temperature simulator\n\n");
    
    int j = 0;              /* Spatial index. */
    static int n = 99;      /* Number of grid points r_j including center and edge points. */
    double dr = (RADIUS_MAX - RADIUS_MIN) / (double) (n + 1);   /* Size of grid point separation. */
    int j_bound = (int) (RADIUS_ROD / dr);      /* This is the value of j where we are at the edge of the rod. */
    
    int k = 0;              /* Temporal index. */
    static int m = 1000;    /* Number of timesteps. */
    double dt = (TIME_MAX - TIME_MIN) / m;      /* Size of timestep. */
    
    double s = (KAPPA * dt) / (pow(dr, 2));     /* This is known as the gain parameter. */
    
    gsl_matrix *Temp = gsl_matrix_alloc(n, 1 + 4);  /* This matrix will hold radius vector and the Temperature vectors for 4 timesteps. */
    for (j = 0; j < n; j++) {
        gsl_matrix_set(Temp, j, 0, (j * dr) + 1);   /* Set the first column(0) of this matrix to store the radius positions 1 to 99. */
    }
    
    /* These vectors will temporarily hold the temperature vectors at even and odd k respectively. */
    gsl_vector *T_even = gsl_vector_alloc(n);
    gsl_vector *T_odd = gsl_vector_alloc(n);
    init_rod_temp_env(T_even);  /* Here the temperature of the rod at k = 0 is initially set to T_ENVIRONMENT in T_even. */

    /* These three vectors contain all the information stored in the tridiagonal matrix A. */
    gsl_vector *diag = gsl_vector_alloc(n);
    gsl_vector *above_diag = gsl_vector_alloc(n - 1);
    gsl_vector *below_diag = gsl_vector_alloc(n - 1);
    make_tridiag_matrix(diag, above_diag, below_diag, n, s);
    
    gsl_vector *source = gsl_vector_alloc(n);   /* This vector will hold the values of Kappa*dt*Source. */
    gsl_vector *b = gsl_vector_alloc(n);        /* This vector will hold the values of b set in the following k loop. */
    
    /* For each timestep k, set the source vector, and then the b vector using the previous temperature and the source vector, and then solves the tridiagonal matrix equation Ax = b. */
    for (k = 1; k <= m; k++) {
        set_source_vector(source, j_bound, dt, k);
        if (k % 2 == 0) {
            set_b_vector(b, T_odd, source, n, s);
            gsl_linalg_solve_tridiag(diag, above_diag, below_diag, b, T_even);
        }
        else {
            set_b_vector(b, T_even, source, n, s);
            gsl_linalg_solve_tridiag(diag, above_diag, below_diag, b, T_odd);
        }
        
        /* Store the Temperature vector after certain time periods in columns of the matrix Temp. */
        if (k * dt == 1) {
            gsl_matrix_set_col(Temp, 1, T_even);
        }
        if (k * dt == 10) {
            gsl_matrix_set_col(Temp, 2, T_even);
        }
        if (k * dt == 50) {
            gsl_matrix_set_col(Temp, 3, T_even);
        }
        if (k * dt == 100) {
            gsl_matrix_set_col(Temp, 4, T_even);
        }
    }
    
    matrix_print(n, 5, Temp);
    plot_data();
    
    gsl_matrix_free(Temp);
    gsl_vector_free(T_even);
    gsl_vector_free(T_odd);
    gsl_vector_free(diag);
    gsl_vector_free(above_diag);
    gsl_vector_free(below_diag);
    gsl_vector_free(source);
    gsl_vector_free(b);
    
    return 0;
}
