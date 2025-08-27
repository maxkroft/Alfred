#include <math.h>
#include <stdio.h>
#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif

int main(void){
    return 0;
}

double findRoot(const double e, const double t, const double tp, const double p){

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
    //par = [p, tc, k, e, w]

    double fc = M_PI/2 - par[4];
    if (fc < 0){
        fc += 2*M_PI;
    }

    double Ec = atan2(sqrt(1 - par[3]*par[3]) * sin(fc), par[3] + cos(fc));

    double tp = par[1] - par[0]/(2*M_PI) * (Ec - par[3] * sin(Ec));

    int i;
    double E;
    double f;

    for (i=0;i<n;i++){

        E = findRoot(par[3], t[i], tp, par[0]);

        f = atan2(sqrt(1 - par[3]*par[3]) * sin(E), cos(E) - par[3]);

        rv[i] = par[2] * (cos(par[4] + f) + par[3] * cos(par[4]));

    }

}


