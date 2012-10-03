//particle.h
#ifndef PARTICLE
#define PARTICLE

//template <typename T, int DIM>
class Particle{

	public:
                float *value;
                float weight;
                int dim;

                Particle();
                Particle(int);

                ~Particle();

                Particle(const Particle&);
                Particle &operator=(const Particle&);
                bool operator < (const Particle & ) const;
};

#endif
