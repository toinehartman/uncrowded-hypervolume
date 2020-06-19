#pragma once

/*

Implementation by T.J.B. Hartman, 2020
*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"

namespace hicam
{

  class BiasVarianceTradeoff_t : public fitness_t
  {

  public:

  	// param
  	size_t K;

  	// (reference) data
  	vec_t xs;
  	vec_t ys;
  	vec_t ys_ref;

    BiasVarianceTradeoff_t(size_t K)
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 3 * K; // default, can be adapted

      this->K = K;

      // dataset
	    double xs_raw[] = {-2.13876, -0.843971, -2.77733, 0.969594, 0.948479, 0.0596248, -2.81538, -2.68752, -2.66449, 1.43964, -1.82261, 0.75355, 1.77127, 1.50504, 0.563664, 0.934696, -0.964971, -0.0580217, -1.03172, 1.74479, 0.949619, -0.235184, -1.03222, -2.34456, -1.96341, 1.3474, -1.99459, -0.616371, 1.23675, -0.115709, -0.620406, -0.313958, -2.91901, -0.51485, -1.23434, 1.8188, -2.15125, 1.94123, -1.34461, 1.81213, -2.54684, -2.63072, 0.402378, -2.42504, 0.166079, -1.07568, -1.61749, -2.91959, -1.71093, 1.08362};
	    double ys_raw[] = {-10.4758, -4.80274, -23.3352, -1.80149, 5.61443, 1.73721, -33.2348, -32.3649, -27.0149, 14.9916, -9.4299, -12.8493, 12.8735, 2.71406, 3.78351, 2.43841, -6.41437, -10.7886, -0.705705, 3.52402, 10.0806, -9.5326, -0.447625, -14.892, -11.4165, 8.8522, -7.94916, 0.822361, 2.71519, 3.67672, 0.177861, -2.70773, -32.1672, -7.76417, -7.15321, 6.14037, -21.8601, -1.02493, -2.55413, 16.949, -23.3274, -29.4496, 12.4751, -18.4906, 1.72832, 1.86347, -2.40768, -27.5401, -1.73747, -1.90976};
	    double ys_ref_raw[] = {-12.4221, -1.94512, -24.7006, 1.38112, 1.30174, -0.440163, -25.6311, -22.5989, -22.081, 3.92336, -8.37716, 0.681445, 6.82847, 4.41418, 0.242749, 1.2513, -2.36352, -0.558217, -2.62995, 6.55641, 1.30596, -0.748192, -2.63202, -15.7325, -10.0323, 3.29358, -10.4298, -1.35054, 2.62841, -0.617258, -1.3592, -0.844905, -28.2908, -1.15132, -3.61498, 7.33543, -12.607, 8.75652, -4.27563, 7.26287, -19.5667, -21.3372, -0.0324742, -17.1864, -0.32934, -2.82032, -6.34927, -28.3062, -7.21931, 1.85605};

	    xs.setRaw(xs_raw, 50);
	    ys.setRaw(ys_raw, 50);
	    ys_ref.setRaw(ys_ref_raw, 50);

      hypervolume_max_f0 = 200;
      hypervolume_max_f1 = 200;

    }
    ~BiasVarianceTradeoff_t() {}

    // number of objectives
    void set_number_of_objectives(size_t &number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
    }

    // any positive value
    void set_number_of_parameters(size_t &number_of_parameters)
    {
      this->number_of_parameters = 3 * K; // just ignore the set value
      number_of_parameters = this->number_of_parameters;
    }

    void get_param_bounds(vec_t &lower, vec_t &upper) const
    {

      lower.clear();
      lower.resize(number_of_parameters, -100);

      upper.clear();
      upper.resize(number_of_parameters, 100);

    }

    double x(size_t i) {
    	return xs[i];
    }

    double y(size_t i) {
    	return ys[i];
    }

    double y_ref(size_t i) {
      return ys_ref[i];
    }

    double weight(size_t k, solution_t &sol) {
	return sol.param[3 * k];
    }

    double mu(size_t k, solution_t &sol) {
	return sol.param[3 * k + 1];
    }

    double sigma(size_t k, solution_t &sol) {
	return sol.param[3 * k + 2];
    }

    double _rbf(double x, double mu, double sigma) {
    	return exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma));
    }

    double r(double x, solution_t &theta) {
    	double val = 0.0;
    	for (size_t k = 0; k < K; k++) {
    		val += weight(k, theta) * _rbf(x, mu(k, theta), sigma(k, theta));
    	}
    	return val;
    }

    void define_problem_evaluation(solution_t &sol)
    {
    	/**
    	 * structure of parameters:
    	 * for K in {0, 3, 6, ...}
    	 * 	w_K 		= sol.param[K]
    	 * 	mu_K 		= sol.param[K + 1]
    	 * 	sigma_K = sol.param[K + 2]
    	 */

    	// copied from other problem
	    sol.constraint = 0;

    	// f0 : goodness of fit (bias?)
    	double bias = 0.f;
	for (size_t i = 0; i < xs.size(); i++) {
		double r_val = r(x(i), sol);
		bias += pow(abs(y(i) - r_val), 2);
    	}
	sol.obj[0] = bias / xs.size();

	    // f1 : smoothness of the obtained result (variance?)
	    double variance = 0.f;
	    for (size_t i = 0; i < K; i++) {
	    	variance += pow(abs(weight(i, sol)), 2);
	    }
	    sol.obj[1] = variance / K;
    }

    std::string name() const
    {
    	std::ostringstream ss;
	    ss << "BiasVarianceTradeoff(K=" << K << ")";
      return ss.str();
    }

  };
}
