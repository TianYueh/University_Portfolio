#include "integrator.h"

#include "configs.h"

void ExplicitEuler::integrate(const std::vector<Particles *> &particles, std::function<void(void)>) const {
  // TODO: Integrate velocity and acceleration
  //   1. Integrate velocity.
  //   2. Integrate acceleration.
  //   3. You should not compute position using acceleration. Since some part only update velocity. (e.g. impulse)
  // Note:
  //   1. You don't need the simulation function in explicit euler.
  //   2. You should do this first because it is very simple. Then you can chech your collision is correct or not.
  //   3. This can be done in 5 lines. (Hint: You can add / multiply all particles at once since it is a large matrix.)
  for (const auto& p : particles) {
    //V = V + a * dt
    p->velocity() += p->acceleration() * deltaTime;
	p->position() += p->velocity() * deltaTime;
  }
}

void ImplicitEuler::integrate(const std::vector<Particles *> &particles,
                              std::function<void(void)> simulateOneStep) const {
  // TODO: Integrate velocity and acceleration
  //   1. Backup original particles' data.
  //   2. Integrate velocity and acceleration using explicit euler to get Xn+1.
  //   3. Compute refined Xn+1 using (1.) and (2.).
  // Note:
  //   1. Use simulateOneStep with modified position and velocity to get Xn+1.

    for (const auto &p : particles) {
        auto curPos = p->position();
        auto curV = p->velocity();
        auto cura = p->acceleration();

        //Compute Xn+1
        p->velocity() += cura * deltaTime;
        p->position() += curV * deltaTime;
        simulateOneStep();
        p->velocity() = curV + cura * deltaTime;
        p->position() = curPos + p->velocity() * deltaTime;

    } 

}

void MidpointEuler::integrate(const std::vector<Particles *> &particles,
                              std::function<void(void)> simulateOneStep) const {
  // TODO: Integrate velocity and acceleration
  //   1. Backup original particles' data.
  //   2. Integrate velocity and acceleration using explicit euler to get Xn+1.
  //   3. Compute refined Xn+1 using (1.) and (2.).
  // Note:
  //   1. Use simulateOneStep with modified position and velocity to get Xn+1.
    for (const auto &p : particles) {
    
    	auto curPos = p->position();
    	auto curV = p->velocity();
    	auto cura = p->acceleration();
    
    	//Compute Xn+1
    	p->velocity() += cura * deltaTime / 2;
    	p->position() += curV * deltaTime / 2;
    	simulateOneStep();
    	p->velocity() = curV + cura * deltaTime;
    	p->position() = curPos + p->velocity() * deltaTime; 
    } 

}

void RungeKuttaFourth::integrate(const std::vector<Particles *> &particles,
                                 std::function<void(void)> simulateOneStep) const {
  // TODO: Integrate velocity and acceleration
  //   1. Backup original particles' data.
  //   2. Compute k1, k2, k3, k4
  //   3. Compute refined Xn+1 using (1.) and (2.).
  // Note:
  //   1. Use simulateOneStep with modified position and velocity to get Xn+1.

    for (const auto &p : particles) {
        auto curPos = p->position();
		auto curV = p->velocity();
		auto cura = p->acceleration();
		//Compute K1
        auto k1v = cura*deltaTime;
        auto k1Pos = curV*deltaTime;
        //Compute K2
        p->velocity() += cura * deltaTime / 2;
        p->position() += k1Pos / 2;
        simulateOneStep();
        auto k2v = curV + p->acceleration() * deltaTime;
        auto k2Pos = k2v * deltaTime;
        p->velocity() = curV;
        p->position() = curPos;
        //Compute K3
        p->velocity() += p->acceleration() * deltaTime / 2;
        p->position() += k2Pos / 2;
        simulateOneStep();
        auto k3v = curV + p->acceleration() * deltaTime;
        auto k3Pos = k3v * deltaTime;
        p->velocity() = curV;
        p->position() = curPos;
        //Compute K4
        p->velocity() += p->acceleration() * deltaTime;
        p->position() += k3Pos;
        simulateOneStep();
        auto k4v = curV + p->acceleration() * deltaTime;
        auto k4Pos = k4v * deltaTime;
        //Compute Xn+1
        auto newPos = curPos + (k1Pos + 2 * k2Pos + 2 * k3Pos + k4Pos) / 6;
        p->position() = newPos;
        p->velocity() = curV + p->acceleration() * deltaTime;
    }
}
