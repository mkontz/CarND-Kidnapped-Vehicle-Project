/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 100;  // TODO: Set the number of particles

  // Clear and reserve
  particles.clear();
  particles.reserve(num_particles);
  weights.resize(num_particles, 1.0);

  // random number generator
  std::default_random_engine gen;

    // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_th(theta, std[2]);

  Particle tmp;
  tmp.weight = 1.0;
  tmp.associations.clear();
  tmp.sense_x.clear();
  tmp.sense_y.clear();
  for (int k = 0; k < num_particles; ++k)
  {
    tmp.id = k;
    tmp.x = dist_x(gen);
    tmp.y = dist_y(gen);
    tmp.theta = dist_th(gen);
    particles.push_back(tmp);
  }

  // set flag indicating that particle is initialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  using std::cos;
  using std::sin;

  // random number generator
  std::default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_th(0.0, std_pos[2]);

  double th1, th2;
  for (int k = 0; k < num_particles; ++k)
  {
    th1 = particles[k].theta;
    th2 = th1 + delta_t * yaw_rate;

    if ( 0.001 < std::abs(yaw_rate))
    {
      particles[k].x += velocity / yaw_rate * (sin(th2) - sin(th1)) + dist_x(gen);
      particles[k].y += velocity / yaw_rate * (cos(th1) - cos(th2)) + dist_y(gen);
    }
    else
    {
      // Avoid divide by zero...since yaw-rate is small
      particles[k].x += cos(th1) * velocity * delta_t + dist_x(gen);
      particles[k].y += sin(th1) * velocity * delta_t + dist_y(gen);
    }

    particles[k].theta = th2 + dist_th(gen);
  }
}

void ParticleFilter::dataAssociation(Particle& particle,
                                     const std::vector<LandmarkObs>& closeLandMarks,
                                     const std::vector<LandmarkObs>& observations)
{
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;

  // reserver memory
  associations.reserve(observations.size());
  sense_x.reserve(observations.size());
  sense_y.reserve(observations.size());

  // tmp variables
  int best_id;
  double cth = std::cos(particle.theta);
  double sth = std::sin(particle.theta);
  double x_o, y_o, d_x, d_y, error_sq, error_sq_best;

  for (size_t k = 0; k < observations.size(); ++k)
  {
    error_sq_best = 1e40;
    x_o = cth * observations[k].x - sth * observations[k].y + particle.x;
    y_o = sth * observations[k].x + cth * observations[k].y + particle.y;

    for (size_t i = 0; i < closeLandMarks.size(); ++i)
    {
      d_x = x_o - closeLandMarks[i].x;
      d_y = y_o - closeLandMarks[i].y;
      error_sq = d_x * d_x + d_y * d_y;

      if (error_sq < error_sq_best)
      {
        best_id = closeLandMarks[i].id;
        error_sq_best = error_sq;
      }
    }

    associations.push_back(best_id);
    sense_x.push_back(x_o);
    sense_y.push_back(y_o);
  }

  SetAssociations(particle, associations, sense_x, sense_y);
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  std::vector<Particle>::iterator it;
  for (it = particles.begin(); it != particles.end(); ++it)
  {
    // Find "close" map landmarks ()
    vector<LandmarkObs> closeLandMarks = findCloseLandMarks(*it, map_landmarks, sensor_range);

    // Associate observations with best landmarks
    dataAssociation(*it, closeLandMarks, observations);
  }

  // calculate weights
  calcWeights(map_landmarks, std_landmark);
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  // Set of current particles
  std::vector<Particle> new_particles;
  new_particles.reserve(particles.size());

  for (size_t k = 0; k < particles.size(); ++k)
  {
    // randomly samples particles according to weights
    new_particles.push_back(particles[dist(gen)]);
  }

  // replace particles with new set
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::vector<LandmarkObs> ParticleFilter::findCloseLandMarks(const Particle& particle,
                                                            const Map& map_landmarks,
                                                            double sensor_range) const
{
  std::vector<LandmarkObs> close;

  // pre calculations & tmp variables
  LandmarkObs tmp;
  double dx, dy;
  double sqrt_range = 2.25 * sensor_range * sensor_range; // allow error to be 150% of sensor range

  std::vector<Map::single_landmark_s>::const_iterator it;
  for (it = map_landmarks.landmark_list.begin(); it != map_landmarks.landmark_list.end(); ++it)
  {
    dx = it->x_f - particle.x;
    dy = it->y_f - particle.y;
    if (dx * dx + dy * dy <= sqrt_range)
    {
      tmp.x = it->x_f;
      tmp.y = it->y_f;
      tmp.id = it->id_i;
      close.push_back(tmp);
    }
  }

  return close;
}

void ParticleFilter::calcWeights(const Map &map_landmarks,
                                 const double std_landmark[])
{
  // Create hash table for fast look-up of landmarks
  std::unordered_map<int, Map::single_landmark_s> hash;
  for (size_t i = 0; i < map_landmarks.landmark_list.size(); ++i)
  {
    hash[map_landmarks.landmark_list[i].id_i] = map_landmarks.landmark_list[i];
  }

  std::vector<Particle>::iterator it;
  for (it = particles.begin(); it != particles.end(); ++it)
  {
    it->weight = 1.0;

    for (size_t i = 0; i < it->associations.size(); ++i)
    {
      it->weight *= density(hash[it->associations[i]].x_f,
                            hash[it->associations[i]].y_f,
                            it->sense_x[i],
                            it->sense_y[i],
                            std_landmark);
    }
  }

  // Update weight vector
  for (size_t k = 0; k < particles.size(); ++k)
  {
    weights[k] = particles[k].weight;
  }
}

double ParticleFilter::density(double xl, double yl, double xo, double yo, const double std_landmark[]) const
{
  double dx = xo - xl;
  double dy = yo - yl;
  double fxy = std::exp(-0.5 * ( dx * std_landmark[0] * dx + dy * std_landmark[1] * dy));
  fxy /= std::sqrt(4.0 * M_PI * M_PI * std_landmark[0] * std_landmark[1]);

  return fxy;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


