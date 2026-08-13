function kT = SW_IsothComp(T,uT,S,uS,P,uP)
    % SW_IsothComp    Isothermal Compressibilty of Seawater
    %=========================================================================
    % USAGE:  kT = SW_IsothComp(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Isothermal compressibility of seawater in the desalination range
    %   for pressures P_sat < P < 12 MPa is given in [1]. The expression
    %   was correlated to pure water data from IAPWS 95 [2] and, seawater
    %   data from Safarov et al. [3,4].
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
    %   kT = isothermal compressibility [1/MPa]
    %
    %   Note: kT will have the same dimensions as T, S and P
    %
    % VALIDITY: (1) 0 < T < 180 C; 0 < S < 56 g/kg; 0 < P < 12 MPa
    %           (2) 0 < T < 180 C; 56 < S < 160 g/kg; 0 < P < 12 MPa
    %
    % ACCURACY: (1) 3.47%
    %           (2) 13.36%
    %
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
    %   [2] W. Wagner and A. Pruss, J. Phys. Chem. Ref. Data. vol 31,2002.
    %   [3] J. Safarov, S. Berndt, F. Millero, R. Feistel, A. Heintz, and E.
    %       Hassel,Deep Sea Res. Part I Oceanogr. Res. Pap., 2012.
    %   [4] J. Safarov, S. Berndt, F. J. Millero, R. Feistel, A. Heintz, and
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
    if ~isequal((T<0)+(T>180),zeros(size(T)))
        warning('Temperature is out of range for isothermal compressibility function 0<T<180 C');
    end

    if ~isequal((S<0)+(S>160),zeros(size(S)))
        warning('Salinity is out of range for isothermal compressibility function 0<S<160 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for isothermal compressibility function P_sat < P < 12 MPa');
    end

    %% BEGIN

    s = S;
    a = [
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

    b = [
        -1.1077e-06
         5.5584e-09
        -4.2539e-11
         8.3702e-09
    ];

    kT_w = a(1) + a(2)*T + a(3)*T.^2 + a(4)*T.^3 + a(5)*T.^4+ a(6)*T.^5+ P.*(a(7)+a(8)*T+a(9)*T.^3);
    kT_s = b(1)*s + b(2)*s.*T + b(3)*s.*T.^2 + b(4)*s.*P;
    kT   = kT_w + kT_s;

end
