function phi = SW_OsmCoeff(T,uT,S,uS)
    % SW_OsmCoeff    Osmotic coefficient of seawater
    %=========================================================================
    % USAGE:  phi = SW_OsmCoeff(T,uT,S,uS)
    %
    % DESCRIPTION:
    %   The osmotic coefficient is presented for a wide temperature and
    %   salinity range. Sharqawy et al. [1] correlate the osmotic
    %   coefficient data of Bromley et al. [2] due its wide parameter range of
    %   0 - 200 C in temperature and 10 - 120 g/kg in salinity. Banchik et al.
    %   [3] have extended the Bromley correlation from 0-10 g/kg using
    %   Brønsted's equation for temperatures from 0 - 120 C. The calculation
    %   and final expression can be found in the paper by Nayar et al. [4]
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
    %   phi = osmotic coefficient [-]
    %
    %   Note: phi will have the same dimensions as T and S
    %
    % VALIDITY: (1) 0 < T < 200 C; 10 < S < 120 g/kg
    %           (2) 0 < T < 200 C; 0 < S < 10 g/kg
    %
    % ACCURACY: (1) 2.57% (Correlation)
    %           (2) 0.89% (Extrapolated)
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2013-07-15: Leonardo D. Banchik (banchik@mit.edu), MIT
    %               - Extended to 0 g/kg in [3]
    %   2015-07-01: Kishor G. Nayar (kgnayar@mit.edu), MIT and
    %               Adam Weiner (aweiner@mit.edu), MIT
    %               - Converted from EES to MATLAB
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
    %       and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    %   [2] K.S. Pitzer, Thermodynamics 3rd Edition, McGraw Hill, Inc., 1995
    %   [3] M. H. Sharqawy, L. D. Banchik, and J. H. Lienhard V, J. Memb. Sci., 211-219, 2013.
    %   [4] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/)
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
        warning('Temperature is out of range for Osmotic Coefficient function 0<T<200 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for Osmotic Coefficient function 0<S<120 g/kg');
    end

    %% BEGIN

    a = [
         8.9453233003E-01;
         4.1560737424E-04;
        -4.6262121398E-06;
         2.2211195897E-11;
        -1.1445456438E-04;
        -1.4783462366E-06;
        -1.3526263499E-11;
         7.0132355546E-06;
         5.6960486681E-08;
        -2.8624032584E-10;
    ];

    % for S <= 10
        S_eq = 10*ones(size(T)); %Correlation matches function at S_equivalent = 10

        Phi_corr_eq = ...
            a(1 )                   + ...
            a(2 ) * T               + ...
            a(3 ) * T.^2            + ...
            a(4 ) * T.^4            + ...
            a(5 ) * S_eq            + ...
            a(6 ) * T       .* S_eq + ...
            a(7 ) * S_eq    .* T.^3 + ...
            a(8 ) * S_eq.^2         + ...
            a(9 ) * S_eq.^2 .* T    + ...
            a(10) * S_eq.^2 .* T.^2;

        dPhi_corr_eq = ...
            a(5 )                    + ...
            a(6 ) * T                + ...
            a(7 ) * T.^3             + ...
            a(8 ) * S_eq         * 2 + ...
            a(9 ) * S_eq .* T    * 2 + ...
            a(10) * S_eq .* T.^2 * 2 ;

        m_sum_eq = S_eq ./ (1000-S_eq)*31.843280;
        dmds_eq = 31.843280 * (1./(1000-S_eq) + S_eq./(1000-S_eq).^2);

        % Pitzer-Bronsted equation:  Phi_pitzer = 1 - (beta)*m_sum^(0.5) + lambda*m_sum "
        % The 2 constants are determined by equating the F and dF/dS of Bromley's correlation with Pitzer's equation"

        beta = -2*( m_sum_eq.^(-0.5) .* (Phi_corr_eq - 1) - dPhi_corr_eq .* m_sum_eq.^(0.5)./dmds_eq);
        lambda = (Phi_corr_eq + beta.*m_sum_eq.^(0.5) - 1) ./ m_sum_eq;

        m_sum = S ./ (1000-S) * 31.843280; %Define molality as a function of salinity
        phi = 1 - beta.*m_sum.^(0.5) + lambda.*m_sum;

    % for S > 10
        filter = find(S>10);
        phi(filter) = ...
            a(1 )                                 + ...
            a(2 ) * T(filter)                     + ...
            a(3 ) * T(filter).^2                  + ...
            a(4 ) * T(filter).^4                  + ...
            a(5 ) * S(filter)                     + ...
            a(6 ) * T(filter)     .* S(filter)    + ...
            a(7 ) * S(filter)     .* T(filter).^3 + ...
            a(8 ) * S(filter).^2                  + ...
            a(9 ) * S(filter).^2  .* T(filter)    + ...
            a(10) * S(filter).^2  .* T(filter).^2;

    end

