function BPE = SW_BPE(T,uT,S,uS)
    % SW_BPE    Boiling point elevation of seawater
    %=========================================================================
    % USAGE:  BPE = SW_BPE(T,uT,S,uS)
    %
    % DESCRIPTION:
    %   Boiling point elevation of seawater using Eq. (36) given in [1]
    %   which best fit the data of [2].
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
    %   Note: T and S must have the same dimensions
    %
    % OUTPUT:
    %   BPE = boiling point elevation [K]
    %
    %   Note: BPE will have the same dimensions as T and S
    %
    % VALIDITY: 0 < T < 200 C; 0 < S < 120 g/kg
    %
    % ACCURACY: 0.018 K
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, "Thermophysical
    %       properties of seawater: A review of existing correlations and data,"
    %       Desalination and Water Treatment, Vol. 16, pp. 354--380, April 2010.
    %       http://web.mit.edu/seawater/
    %   [2] L. A. Bromley, D. Singh, P. Ray, S. Sridhar, and S. M. Read,
    %       Thermodynamic properties of sea salt solutions, AIChE Journal 20,
    %       326-335, 1974.
    %=========================================================================

    %% CHECK INPUT ARGUMENTS

    % CHECK THAT S&T HAVE SAME SHAPE
    if ~isequal(size(S),size(T))
        error('check_stp: S & T must have same dimensions');
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

    % CHECK THAT S & T ARE WITHIN THE FUNCTION RANGE
    if ~isequal((T<0)+(T>200),zeros(size(T)))
        warning('Temperature is out of range for boiling point elevation function 0<T<200 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for boiling point elevation function 0<S<120 g/kg');
    end

    %% BEGIN

    S = S/1000;

    a = [
        -4.5838530457E-04
         2.8230948284E-01
         1.7945189194E+01
         1.5361752708E-04
         5.2669058133E-02
         6.5604855793E+00
    ];

    A = a(1)*T.^2 + a(2)*T + a(3);
    B = a(4)*T.^2 + a(5)*T + a(6);
    BPE = A.*S.^2 + B.*S;

end
