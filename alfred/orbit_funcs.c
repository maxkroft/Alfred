#include <math.h>
#include <stdio.h>
#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif

int main(void){
    return 0;
}

double solve_Kepler(const double t, const double p, const double tp, const double e){

    int i;
    double f, df, dx, E;

    E = 2*M_PI * ((t - tp) / p);

    for (i=0;i<100;i++){

        f = E - e * sin(E) - 2*M_PI * ((t - tp) / p);
        df = 1 - e * cos(E);
        dx = f/df;
        E -= dx;

        if (fabs(dx) < 1e-9) return E;

    }

    return E;

}


void rvModel(const double par[], const double t[], const int n, double rv[]){

    double p = par[0];
    double tc = par[1];
    double k = par[2];
    double e = par[3];
    double w = par[4];

    double sqe = sqrt(1.0 - e*e);
    double cosw = cos(w);
    
    double fc = M_PI/2 - w;
    if (fc < 0){
        fc += 2*M_PI;
    }
    double Ec = atan2(sqe * sin(fc), e + cos(fc));
    double tp = tc - p/(2*M_PI) * (Ec - e * sin(Ec));

    for (int i=0;i<n;i++){

        double E = solve_Kepler(t[i], p, tp, e);

        double f = atan2(sqe * sin(E), cos(E) - e);

        rv[i] = k * (cos(w + f) + e * cosw);

    }

}


void astrometryModel(const double par[], const double t[], const double psi[], const int n, double x[], double y[], double eta[]){

    double p = par[0]; //days
    double tc = par[1]; //bjd
    double e = par[2];
    double w = par[3]; //radians
    double inc = par[4]; //radians
    double omega = par[5]; //radians
    double mp = par[6]; //msun
    double ms = par[7]; //msun
    double plx = par[8]; //mas

    double q = mp/ms;
    double p_yr = p/365.25;
    double a = cbrt(p_yr*p_yr * (mp + ms));
    double a_star = a * q / (1.0 + q);
    double a_mas = a_star * plx;
    double sqe = sqrt(1.0 - e*e);

    double fc = M_PI/2 - w;
    if (fc < 0){
        fc += 2*M_PI;
    }
    double Ec = atan2(sqe * sin(fc), e + cos(fc));
    double tp = tc - p/(2*M_PI) * (Ec - e * sin(Ec));

    double A = a_mas * (cos(w) * cos(omega) - sin(w) * sin(omega) * cos(inc));
    double B = a_mas * (cos(w) * sin(omega) + sin(w) * cos(omega) * cos(inc));
    double F = a_mas * (-sin(w) * cos(omega) - cos(w) * sin(omega) * cos(inc));
    double G = a_mas * (-sin(w) * sin(omega) + cos(w) * cos(omega) * cos(inc));


    for (int i=0;i<n;i++){

        double E = solve_Kepler(t[i], p, tp, e);

        double X = cos(E) - e;
        double Y = sqe * sin(E);

        x[i] = B*X + G*Y;
        y[i] = A*X + F*Y;
        eta[i] = y[i]*cos(psi[i]) + x[i]*sin(psi[i]);

    }

}