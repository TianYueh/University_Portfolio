%110550085房天越
classdef Vec3
    properties
        %Original properties
        x = 0;
        y = 0;
        z = 0;

    end
    methods
        function obj = Vec3(a, b, c)
            if nargin == 0
                %default
                obj.x = 0;
                obj.y = 0;
                obj.z = 0;
            elseif nargin == 3
                %check if they have the same sizes and initialize them
                if isequal(size(a), size(b), size(c))
                    obj.x = a;
                    obj.y = b;
                    obj.z = c;
                else
                    error('Input must be numerical arrays of identical sizes.')
                end

            elseif nargin ~= 3
                error("Wrong number of argin");
            else 
                error('Some other errors bruh');
            end

        end

        function f = plus(f1, f2)
            if isscalar(f1) && isscalar(f2) %scalar
                f = reshape(Vec3([f1.x]+[f2.x], [f1.y]+[f2.y], [f1.z]+[f2.z]), size(f1));
            elseif ~isscalar(f1) && isscalar(f2) %f2 is scalar
                sz = size(f1);
                for i=1:sz(1)
                    for j=1:sz(2)
                        f(i, j)= reshape(Vec3([f1(i, j).x] + [f2.x],[f1(i, j).y] + [f2.y],[f1(i, j).z] + [f2.z]), size(f1(i, j)));
                    end
                end
            elseif isscalar(f1) && ~isscalar(f2) %f1 is scalar
                sz = size(f2);
                for i=1:sz(1)
                    for j=1:sz(2)
                        f(i, j)= reshape(Vec3([f2(i, j).x] + [f1.x],[f2(i, j).y] + [f1.y],[f2(i, j).z] + [f1.z]), size(f2(i, j)));
                    end
                end
            else %both are not scalars
                sz1 = size(f1);
                sz2 = size(f2);
                if sz1(1)~=sz2(1) | sz1(2)~=sz2(2)
                    error('Expect input to have identical sizes.');
                end
                sz = size(f1);
                for i=1:sz(1)
                    for j=1:sz(2)
                        f(i, j)= reshape(Vec3([f1(i, j).x] + [f2(i, j).x],[f1(i, j).y] + [f2(i, j).y],[f1(i, j).z] + [f2(i, j).z]), size(f1(i, j)));
                    end
                end
            end
        end

        function f = minus(f1, f2)
            if isscalar(f1) && isscalar(f2) %Both are scalar
                f = reshape(Vec3([f1.x]-[f2.x], [f1.y]-[f2.y], [f1.z]-[f2.z]), size(f1));
            elseif ~isscalar(f1) && isscalar(f2) %f2 is scalar
                sz = size(f1);
                for i=1:sz(1)
                    for j=1:sz(2)
                        f(i, j)= reshape(Vec3([f1(i, j).x] - [f2.x],[f1(i, j).y] - [f2.y],[f1(i, j).z] - [f2.z]), size(f1(i, j)));
                    end
                end
            elseif ~isscalar(f1) && ~isscalar(f2) %Both are non-scalar
                sz1 = size(f1);
                sz2 = size(f2);
                if sz1(1)~=sz2(1) | sz1(2)~=sz2(2)
                    error('Expect input to have identical sizes.');
                end
                sz = size(f1);
                for i=1:sz(1)
                    for j=1:sz(2)
                        f(i, j)= reshape(Vec3([f1(i, j).x] - [f2(i, j).x],[f1(i, j).y] - [f2(i, j).y],[f1(i, j).z] - [f2(i, j).z]), size(f1(i, j)));
                    end
                end
            else %f1 is scalar, this cannot be dealt
                error('A scalar Vec3 cannot minus a non-scalar Vec3');
            end
        end

        function f = eq(f1, f2)
            if isscalar(f1) && isscalar(f2) %both scalar
                f = (f1.x == f2.x) & (f1.y == f2.y) & (f1.z == f2.z);
            elseif ~isscalar(f1) && ~isscalar(f2) %both not scalar
                sz1 = size(f1);
                sz2 = size(f2);
                if sz1(1)~=sz2(1) | sz1(2)~=sz2(2)
                    error('Expect input to have identical sizes.');
                end
                sz = size(f1);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        f(i, j) = (f1(i,j).x == f2(i,j).x) & (f1(i,j).y == f2(i,j).y) & (f1(i,j).z == f2(i,j).z);
                    end
                end
            elseif isscalar(f1) && ~isscalar(f2) %f1 is scalar
                sz = size(f2);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        f(i, j) = (f1.x == f2(i,j).x) & (f1.y == f2(i,j).y) & (f1.z == f2(i,j).z);
                    end
                end
                
            elseif ~isscalar(f1) && isscalar(f2) %f2 is scalar
                sz = size(f1);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        f(i, j) = (f2.x == f1(i,j).x) & (f2.y == f1(i,j).y) & (f2.z == f1(i,j).z);
                    end
                end
            else
                error('How do we get there?')
            end
            
        end

        function f = inner_prod(f1, f2)
            if ~isscalar(f1) && ~isscalar(f2) %Check if both are arrays and have different sizes, if so, error
                sz1 = size(f1);
                sz2 = size(f2);
                if sz1(1)~=sz2(1) | sz1(2)~=sz2(2)
                    error('Expect input to have identical sizes or one scalar and one array.');
                end
            end

            if isscalar(f1) && isscalar(f2) %both are scalars
                f = [f1.x]*[f2.x]+[f1.y]*[f2.y]+[f1.z]*[f2.z];

            elseif isscalar(f1) && ~isscalar(f2) % f1 is scalar
                sz = size(f2);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        f(i, j) = f1.x * f2(i, j).x + f1.y * f2(i, j).y + f1.z * f2(i, j).z;
                    end
                end

            elseif ~isscalar(f1) && isscalar(f2) %f2 is scalar
                sz = size(f1);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        f(i, j) = f2.x * f1(i, j).x + f2.y * f1(i, j).y + f2.z * f1(i, j).z;
                    end
                end
                %f = f1.x .* f2.x + f1.y .* f2.y + f1.z .* f2.z;
            
            else %both are not scalar, and checked above they have the same size
                sz = size(f1);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        f(i, j) = f2(i, j).x * f1(i, j).x + f2(i, j).y * f1(i, j).y + f2(i, j).z * f1(i, j).z;
                    end
                end
            end
        end

        function v = norm(p)
            if isscalar(p)
                v = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            else %Not scalar, iterate it
                sz = size(p);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        v(i, j) = p(i, j).x * p(i, j).x + p(i, j).y * p(i, j).y + p(i, j).z * p(i, j).z;
                    end
                end
            end
            
        end

        function disp(ob)
            if isscalar(ob)
                fprintf('(%g,%g,%g)\n', ob.x, ob.y, ob.z);
            else %not scalar, iterate
                sz = size(ob);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        fprintf('(%g,%g,%g) ', ob(i, j).x, ob(i,j).y, ob(i,j).z);
                    end
                    fprintf('\n');
                end
            end
        end

        function f = iszero(p)
            %iterate it, return is a logical array
            sz = size(p);
            for i = 1:sz(1)
                for j = 1:sz(2)
                    f(i, j) = ((p(i, j).x == 0) & (p(i, j).y == 0) & (p(i, j).z == 0));
                end
            end
        end

        function f = normalize(p)
            %fprintf('Hello');
            if isscalar(p) 
                n = p.norm();
                f = Vec3(p.x / n, p.y / n, p.z / n);
                if n==0 %if norm == 0 ,then output NaN for x, y, and, z
                    f.x = NaN;
                    f.y = NaN;
                    f.z = NaN;
                end
            else
                sz = size(p);
                for i = 1:sz(1)
                    for j = 1:sz(2)
                        n = norm(p(i, j));
                        f(i, j) = Vec3(p(i, j).x / n, p(i, j).y / n, p(i, j).z / n);
                        if(n == 0) %if norm == 0 ,then output NaN for x, y, and, z
                            f(i, j).x = NaN;
                            f(i, j).y = NaN;
                            f(i, j).z = NaN;
                        end
                    end
                end
            end
        
            
        end


    end

end