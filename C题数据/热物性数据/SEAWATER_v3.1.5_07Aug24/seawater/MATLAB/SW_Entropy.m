function s = SW_Entropy(T,uT,S,uS,P,uP)
    % SW_Entropy    Specific entropy of seawater
    %=========================================================================
    % USAGE:  s = SW_Entropy(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Specific entropy of seawater at atmospheric pressure (0.1 MPa) using Eq. (46)
    %   given in [1] which best fit the data of [2]. The pure water specific
    %   entropy equation is a best fit to the data of [3].
    %   Entropy at non-atmospheric pressures (P_sat < P < 12 MPa) is obtained
    %   from [4].
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
    %   s = specific entropy [J/kg-K]
    %
    %   Note: s will have the same dimensions as T, S and P
    %
    % VALIDITY: (1) 10 < T < 120 C; S = 0 g/kg; 0 < P < 12 MPa
    %           (2) 10 < T < 80 C; 0 < S < 120 g/kg; P = 0.1 MPa
    %           (3) 10 < T < 40 C; 0 < S < 42 g/kg; 0 < P < 12 MPa
    %           (4) 80 < T < 120 C; 0 < S < 120 g/kg; 0.1 < P = P0 < 1 MPa
    %           (5) 40 < T < 120 C; 0 < S < 42 g/kg; 0 < P < 12 MPa
    %           (6) 10 < T < 120 C; 42 < S < 120 g/kg; 0 < P < 12 MPa
    %
    % ACCURACY: (1) 0.5%
    %           (2) 0.5%
    %           (3) 0.5%
    %           (4) 1.47% (Extrapolated)
    %           (5) 1.47% (Extrapolated)
    %           (6) 1.47% (Extrapolated)
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-04-15: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Extended function to 12 MPa
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
    %       and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    %   [2]  IAPWS release on the thermodynamic properties of seawater, 2008
    %   [3]  IAPWS release on the Thermodynamic properties of ordinary water substance, 1996.
    %   [4]  K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/) 
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
    if ~isequal((T<10)+(T>120),zeros(size(T)))
        warning('Temperature is out of range for entropy function 10<T<120 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for entropy function 0<S<120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for entropy function P_sat < P < 12 MPa');
    end

    %% BEGIN

    P0               = 0.101325*ones(size(T));
    P0(find(T>=100)) = SW_Psat(T(find(T>=100)),'C',S(find(T>=100)),'ppt')/1E6;


    S_gkg=S;
    S = S/1000;

    a = [
         1.5432265080E-01
         1.5382700241E+01
        -2.9963211781E-02
         8.1929151062E-05
        -1.3699640311E-07
    ];

    s_w = a(1) + a(2)*T + a(3)*T.^2 + a(4)*T.^3 + a(5)*T.^4;

    b = [
        -4.2307343871E+02
         1.4630334922E+04
        -9.8796297642E+04
         3.0946224962E+05
         2.5623880831E+01
        -1.4432346624E-01
         5.8790568541E-04
        -6.1106764270E+01
         8.0408001971E+01
         3.0354282687E-01
    ];

    s_sw_P0 = s_w - S.*(b(1) + b(2)*S + b(3)*S.^2 + b(4)*S.^3 + b(5)*T + b(6)*T.^2 + b(7)*T.^3 + b(8)*S.*T + b(9)*S.^2.*T + b(10)*S.*T.^2);

    c = [-4.4786 * 10 ^ (-3)
    -1.1654 * 10 ^ (-2)
    6.1154 * 10 ^ (-5)
   -2.0696 * 10 ^ (-7)
    -1.5531 * 10 ^ (-3)
    4.0054 * 10 ^ (-5)
    -1.4193 * 10 ^ (-7)
    3.3142 * 10 ^ (-10)];

    s_P = (P - P0).*(c(1) + c(2)*T + c(3)*T.^2 + c(4)*T.^3 + S_gkg.*(c(5) + c(6).*T + c(7)*T.^2 + c(8)*T.^3));

    s=s_sw_P0+s_P;
end
