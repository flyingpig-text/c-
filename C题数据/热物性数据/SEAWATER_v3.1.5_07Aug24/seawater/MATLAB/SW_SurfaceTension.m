function sigma = SW_SurfaceTension(T,uT,S,uS)
    % SW_SurfaceTension    Surface tension of seawater
    %=========================================================================
    % USAGE:  sigma = SW_SurfaceTension(T,uT,S,uS)
    %
    % DESCRIPTION:
    %   Surface tension of seawater is presented for a wide temperature and
    %   salinity range. Surface tension was measured by Nayar et al. [1] for a
    %   temperature range of 1 - 92 C and salinities 0 - 131 g/kg.
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
    %   sigma = surface tension [mN/m]
    %
    %   Note: sigma will have the same dimensions as T and S
    %
    % VALIDITY: (1) 1 < T < 92 C; 0 < S < 131 g/kg; 0.1 < P = P0 < 1 MPa
    %           (2) 1 < T < 100 C; 0 < S < 131 g/kg; 0.1 < P = P0 < 1 MPa
    %
    % ACCURACY: (1) 0.60%
    %           (2) 0.60% (Extrapolated)
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2014-11-20: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - New correlation based on experiments [1]
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %   2016-04-14: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Corrected range of validity of surface tension function     
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %  [1] K. G. Nayar, D. Panchanathan, G. H. Mckinley, and J. H. Lienhard V,
    %      Journal of Physical and Chemical Reference Data, 43(4), 2014.
    %  [2] IAPWS, Release on the Surface Tension of Ordinary Water Substance, 1994.
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
    if ~isequal((T<0)+(T>100),zeros(size(T)))
        warning('Temperature is out of range for surface tension function 0<T<100 C');
    end

    if ~isequal((S<0)+(S>131),zeros(size(S)))
        warning('Salinity is out of range for surface tension function 0<S<131 g/kg');
    end

    %% BEGIN

    sigma_w = 235.8*((1-((T+273.15)/647.096)).^1.256).*(1-0.625*(1-((T+273.15)/647.096)));

    a = [3.766E-04
        2.347E-06];

    sigma = sigma_w.*(1+(a(1)*S+a(2)*S.*T));

end
