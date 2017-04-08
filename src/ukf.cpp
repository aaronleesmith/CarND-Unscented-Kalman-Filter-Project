#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#pragma clean diagnostic push
#pragma ide diagnostic ignored "IncompatibleTypes"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  previous_timestamp_ = 0.0;

  n_x_ = 5;

  n_aug_ = 7;

  n_sigma_ = 2 * n_aug_ + 1;

  n_z_ = 3;

  // todo: look into better lambda values.
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  x_aug_ = VectorXd(n_aug_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0,    0,    0,
        0, 1, 0,    0,    0,
        0, 0, 1000, 0,    0,
        0, 0, 0,    1000, 0,
        0, 0, 0,    0,    1;

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  Xsig_aug_ = MatrixXd(n_aug_, n_sigma_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = .3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = .25;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.25;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.06;

  weights_ = VectorXd(n_sigma_);

  n_z_laser_ = 2;

  n_z_radar_ = 3;

  R_lidar_ = MatrixXd::Zero(n_z_laser_, n_z_laser_);
  R_lidar_ << pow(std_laspx_, 2), 0,
              0,                  pow(std_laspy_, 2);

  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  R_radar_ << pow(std_radr_, 2),  0,                    0,
              0,                  pow(std_radphi_, 2),  0,
              0,                  0,                    pow(std_radrd_, 2);

}

UKF::~UKF() {}

void UKF::Init(MeasurementPackage measurement_pack) {
  if (is_initialized_) {
    return;
  }

  double px, py, vx, vy, velocity, yaw, yaw_rate;

  /**
   * Initialize x based on initial measurement.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    double range = measurement_pack.raw_measurements_[0];
    double bearing = measurement_pack.raw_measurements_[1];
    double range_rate = measurement_pack.raw_measurements_[2];

    px = range * cos(bearing);
    py = range * sin(bearing);
    vx = range_rate * cos(bearing);
    vy = range_rate * sin(bearing);
    velocity = sqrt(pow(vx, 2) * pow(vy, 2));
    yaw = vx == 0 ? 0 : vy / vx;
    yaw_rate = 0;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    /**
    Initialize state.
    */
    px = measurement_pack.raw_measurements_[0];
    py = measurement_pack.raw_measurements_[1];
    velocity = 0;
    yaw = 0;
    yaw_rate = 0;
  }

  // Initialize the state vector.
  x_ << px, py, velocity, yaw, yaw_rate;

  // Initialize the weights vector
  GenerateSigmaWeights();

  previous_timestamp_ = measurement_pack.timestamp_;
  is_initialized_ = true;
  return;
}

double UKF::CalculateDt(double new_timestamp) {
  double dt = (new_timestamp - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  return dt;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Initialization
  Init(meas_package);

  // Prediction
  Prediction(CalculateDt(meas_package.timestamp_));

  // Update
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else {
    std::cout << "Update step has been skipped because either radar or laser measurements are disabled.";
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  MatrixXd Zsig = MatrixXd(n_z_laser_, n_sigma_);
  MatrixXd S = MatrixXd(n_z_laser_, n_z_laser_);
  VectorXd z_pred = VectorXd(n_z_laser_);
  MatrixXd R = MatrixXd(n_z_laser_, n_z_laser_);
  MatrixXd Tc = MatrixXd(n_x_, n_z_laser_);

  Zsig.row(0) = Xsig_pred_.row(0);
  Zsig.row(1) = Xsig_pred_.row(1);

  //mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2 * n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix

  R <<    pow(std_laspx_, 2), 0,
          0,                  pow(std_laspy_, 2);
  S = S + R;

  // Update state
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = VectorXd(n_z_laser_);
  z <<  meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1];
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_sigma_);
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  VectorXd z_pred = VectorXd(n_z_radar_);
  MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2 * n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix

  R <<    pow(std_radr_, 2), 0,                  0,
          0,                pow(std_radphi_, 2), 0,
          0,                0,                  pow(std_radrd_, 2);
  S = S + R;

  // Update state
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = VectorXd(n_z_radar_);
  z <<  meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1],
        meas_package.raw_measurements_[2];
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Update NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}

void UKF::GenerateSigmaWeights() {
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_[0] = lambda_ / (lambda_ + n_aug_);
}

void UKF::GenerateAugmentedSigmaPoints() {

  // Create augmented matrices.
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd A;
  // Create process noise covariance matrix.
  MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);

  // Populate matrices.
  Q << pow(std_a_, 2), 0,
      0,              pow(std_yawdd_, 2);

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;
  A = P_aug.llt().matrixL();

  x_aug << x_, 0, 0;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
}

void UKF::PredictSigmaPoints(double dt) {
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
    }
    else {
      px_p = p_x + v*dt*cos(yaw);
      py_p = p_y + v*dt*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*dt;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
    py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
    v_p = v_p + nu_a*dt;

    yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
    yawd_p = yawd_p + nu_yawdd*dt;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {
  //predicted state mean
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}