#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // The last timestamp, used to calculate dt.
  long long previous_timestamp_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Number of sigma points.
  int n_sigma_;

  ///* Number of measurement points.
  int n_z_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* Augmented state vector (incorporates non-linear noise).
  VectorXd x_aug_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* Augmented sigma points matrix
  MatrixXd Xsig_aug_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  VectorXd weights;

  // Size of the observation vector for lidar.
  int n_z_laser_;

  // Size of the observation vector for radar.
  int n_z_radar_;

  // Measurement noise for lidar.
  MatrixXd R_lidar_;

  // Measurement noise for radar.
  MatrixXd R_radar_;
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
  void GenerateAugmentedSigmaPoints();
  void SigmaPointPrediction();
  void PredictMeanAndCovariance();
  void PredictRadarMeasurement();
  void PredictSigmaPoints(double dt);

  /**
   * Updates the state given the measruement vector z, z sigma points, predicted measurement S.
   * @param z
   * @param Zsig
   * @param z_pred
   * @param S
   */
  void UpdateState(VectorXd& z, MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S);

  /**
   * Generates sigma weights based on class level n_x_ and n_sigma_ values.
   */
  void GenerateSigmaWeights();

  /**
   * Initializes variables for the prediction - update steps to follow.
   */
  void Init(MeasurementPackage meas_package);

  // todo: shuold this be a long long?
  double CalculateDt(double new_timestamp);

};

#endif /* UKF_H */
