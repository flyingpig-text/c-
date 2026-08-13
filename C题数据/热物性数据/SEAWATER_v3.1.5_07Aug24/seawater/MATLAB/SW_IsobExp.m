function betaP_sw = SW_IsobExp(T,uT,S,uS,P,uP)
    % SW_IsobExp    Isobaric Expansivity of Seawater
    %=========================================================================
    % USAGE:  kT = SW_IsothComp(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Isobaric thermal expansivity of seawater in the desalination range
    %   for pressures P_sat < P < 12 MPa is given in [1]. The expression is
    %   obtained using the seawater density function in [2], pure water data
    %   from IAPWS 95 [3] and, seawater data from Safarov et al. [4,5].
    %
    % INPUT:
    %   T  = temperature
    %   uT = temperature unit
    %        'C'  : [degree Celsius] (ITS-90)
    %        'K'  : [Kelvin]
    %        'F'  : [degree Fahrenheit]
    %        'R'  : [Rankine]
    %   S  = salinity
    %   uS = salinity unit
    %        'ppt': [g/kg]  (reference-composition salinity)
    %        'ppm': [mg/kg] (in parts per million)
    %        'w'  : [kg/kg] (mass fraction)
    %        '%'  : [kg/kg] (in parts per hundred)
    %
    %   P  = pressure
    %   uP = pressure unit
    %        'MPa': [MPa]
    %        'bar': [bar]
    %        'kPa': [kPa]
    %        'Pa' : [Pa]
    %
    %   Note: T, S and P must have the same dimensions
    %
    % OUTPUT:
    %   bP = isobaric expansivity [1/K]
    %
    %   Note: bP will have the same dimensions as T, S and P
    %
    % VALIDITY: (1) 10 < T < 180 C; S = 0 g/kg; 0 < P < 12 MPa
    %           (2) 10 < T < 180 C; 0 < S < 56 g/kg; 0.2 < P < 12 MPa
    %           (3) 10 < T < 180 C; 56 < S < 150 g/kg; 0 < P < 12 MPa
    %
    % ACCURACY: (1) 11.37%
    %           (2) 11.37%
    %           (3) 18.30% (Extrapolation)
    %
    % REVISION HISTORY:
    %   2014-07-30: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %           - Initial version based on correlation from [1]
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and license.
    %
    % REFERENCES:
    %   [1] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/) 
    %   [2] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
    %       and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    %   [3] W. Wagner and A. Pruss, J. Phys. Chem. Ref. Data. vol 31,2002.
    %   [4] J. Safarov, S. Berndt, F. Millero, R. Feistel, A. Heintz, and E.
    %       Hassel,Deep Sea Res. Part I Oceanogr. Res. Pap., 2012.
    %   [5] J. Safarov, S. Berndt, F. J. Millero, R. Feistel, A. Heintz, and
    %       E. P. Hassel, Deep Sea Res. Part I Oceanogr. Res. Pap., 2013.
    %=========================================================================

    %% CHECK INPUT ARGUMENTS
    %
    % CHECK THAT S,T & P HAVE SAME SHAPE
    if ~isequal(size(S),size(T), size(P))
        error('check_stp: S, T & P must have same dimensions');
    end

    % CONVERT TEMPERATURE INPUT TO °C
    switch lower(uT)
        case 'c'
        case 'k'
            T = T - 273.15;
        case 'f'
            T = 5/9*(T-32);
        case 'r'
            T = 5/9*(T-491.67);
        otherwise
            error('Not a recognized temperature unit. Please use ''C'', ''K'', ''F'', or ''R''');
    end

    % CONVERT SALINITY TO PPT
    switch lower(uS)
        case 'ppt'
        case 'ppm'
            S = S/1000;
        case 'w'
            S = S*1000;
        case '%'
            S = S*10;
        otherwise
            error('Not a recognized salinity unit. Please use ''ppt'', ''ppm'', ''w'', or ''%''');
    end

    % CONVERT PRESSURE INPUT TO MPa
    switch lower(uP)
        case 'mpa'
        case 'bar'
            P = P/10;
        case 'kpa'
            P = P/1000;
        case 'pa'
            P = P/1E6;
        otherwise
            error('Not a recognized pressure unit. Please use ''MPa'', ''bar'', ''kPa'', or ''Pa''');
    end

    % CHECK THAT S, T & P ARE WITHIN THE FUNCTION RANGE
    if ~isequal((T<10)+(T>180),zeros(size(T)))
        warning('Temperature is out of range for isobaric expansivity function 10<T<180 C');
    end

    if ~isequal((S<0)+(S>150),zeros(size(S)))
        warning('Salinity is out of range for isobaric expansivity function 0<S<150 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for isobaric expansivity function P_sat < P < 12 MPa');
    end


    %% BEGIN

    P0 = Psat;
    P0(find(T<100)) = 0.101325;

    s = S/1000;

    a = [
         9.9992293295E+02
         2.0341179217E-02
        -6.1624591598E-03
         2.2614664708E-05
        -4.6570659168E-08
    ];

    b = [
         8.0200240891E+02
        -2.0005183488E+00
         1.6771024982E-02
        -3.0600536746E-05
        -1.6132224742E-05
    ];

    rho_w = a(1) + a(2)*T + a(3)*T.^2 + a(4)*T.^3 + a(5)*T.^4;
    D_rho = b(1)*s + b(2)*s.*T + b(3)*s.*T.^2 + b(4)*s.*T.^3 + b(5)*s.^2.*T.^2;
    rho_sw_sharq   = rho_w + D_rho;

    drho_wdT = a(2) + 2*a(3)*T+ 3*a(4)*T.^2 + 4*a(5)*T.^ 3;
    dD_rhodT = b(2)*s + 2*b(3)*s.*T + 3*b(4)*s.*T.^2 + 2*b(5)*s.^2.*T.^1;
    drho_sw_sharqdT = drho_wdT + dD_rhodT;


    c = [
         5.0792E-04
        -3.4168E-06
         5.6931E-08
        -3.7263E-10
         1.4465E-12
        -1.7058E-15
        -1.3389E-06
         4.8603E-09
        -6.8039E-13
    ];

    d = [
        -1.1077e-06
         5.5584e-09
        -4.2539e-11
         8.3702e-09
    ];

    kT = c(1) + c(2)*T + c(3)*T.^2 + c(4)*T.^3 + c(5)*T.^4 + c(6)*T.^5 + ...
        P.*(c(7) + c(8)*T + c(9)*T.^3) + ...
        S.*(d(1) + d(2)*T + d(3)*T.^2 + d(4)*P);

    F_P = exp(...
        (P-P0).*(c(1) + c(2)*T + c(3)*T.^2 + c(4)*T.^3 + c(5)*T.^4 + c(6)*T.^5 + S.*(d(1) + d(2)*T + d(3)*T.^2)) + ...
        0.5*(P.^2-P0.^2).*(c(7) + c(8)*T + c(9)*T.^3 + d(4)*S) ...
    );

    dF_PdT = F_P.*(...
        (P-P0).*(c(2) + 2*c(3)*T.^1 + 3*c(4)*T.^2 + 4*c(5)*T.^3 + 5*c(6)*T.^4 + S.*(d(2) + 2*d(3)*T.^1)) + ...
        0.5*(P.^2-P0.^2).*(c(8) + 3*c(9)*T.^2) ...
    );

    rho_sw   = rho_sw_sharq.*F_P;
    drho_dT  = drho_sw_sharqdT.*F_P+rho_sw_sharq.*dF_PdT;
    betaP_sw = -drho_dT./rho_sw;

end
