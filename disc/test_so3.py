%load_ext autoreload
%autoreload 2
import numpy as np
import lorm
import energy_stippling_so3
import energy_curveling_so3
import matplotlib.pyplot as pl
%matplotlib inline

so3 = lorm.manif.SO3()
print(so3._description)

points = lorm.manif.ManifoldPointArrayParameterized(so3)
M = 200
#points.coords = np.random.randn(M,9)
local_coords = np.zeros([M,3])
for i in range(M):
    local_coords[i,0] = 2*np.pi*(i+0.5)/M+(0.1*np.sin(20*np.pi*i/M))
    local_coords[i,1] = np.arccos(0.7)#np.sin(np.pi*(i+0.5)/M)
    local_coords[i,2] = 2*np.pi*(i+0.5)/M

points.local_coords = local_coords
points.coords = coords
N = 5
#disc = energy_stippling_so3.plan(M,N)
#alpha = 0.1 N = 5
#alpha = 0.0001 N = 15
disc = energy_curveling_so3.plan(M,N,alpha=0.1,power=2)
for n in range(1,N+1):
    disc._lambda_hat[n,:,:] = 1./(2.*n+1)**4

# legendre_p_n(1)
#int_pn = 0.5*8*np.pi**2 *np.ones(51)
# legendre_p_n(-1)
#int_pn = 0.3* 8*np.pi**2 *np.array([(-1)**n for n in range(51)])
# legendre_p_n(-0.95)
#int_pn = 8*np.pi**2 *np.array([1, -0.95, 0.85375, -0.718438, 0.55409, -0.372744, 0.187454,-0.0112272, -0.144023, 0.268422, -0.35488, 0.399604, -0.402305, 0.366116, -0.297207, 0.204162, -0.0971544, -0.0129884, 0.115749, -0.201832, 0.263931])
# legendre_p_n(1/2)
int_pn = 8*np.pi**2 *np.array([1.0,0.5,-0.125,-0.4375,-0.2890625,0.08984375,0.3232421875,0.22314453125,-0.073638916015625,-0.2678985595703125,-0.1882286071777344,0.06387138366699219,0.233752965927124,0.1658042669296265,-0.05717363953590393,-0.2100185006856918,-0.149855135474354,0.05221683974377811,0.1922962221433409,0.1377672102244105,-0.04835838106737356,-0.1784138579223509,-0.1281987246779863,0.04524493778421856,0.1671594460634509,0.1203811168693321,-0.04266414120839634,-0.1577966215046347,-0.1138384028125269,0.04047968696457777,0.1498488149006108,0.1082580079203644,-0.038599562888358,-0.1429921832522211,-0.103425075400988,0.03695911797832659,0.1369979535351437,0.09918640694089159,-0.03551142159254956,-0.1316993126940266,-0.09542943523261548,0.03422147270341801,0.1269713800411686,0.09206934414432813,-0.03306256526118129,-0.1227185621438446,-0.0890408508867776,0.03201392111450407,0.1188662759295311,0.08629277896094001,-0.03105909923960982])
# legendre_p_n(2/3)
#int_pn += 0.8*np.array([1.0,0.6666666666666666,0.1666666666666667,-0.2592592592592592,-0.4274691358024691,-0.3055555555555556,-0.01723251028806584,0.2405692729766804,0.3157900377229081,0.1838221752273536,-0.05136961199596945,-0.2324905745651916,-0.249982478725884,-0.1058836731268538,0.09599043622522302,0.2225457682753511,0.1974639167278485,0.04608669856512062,-0.1267516824733672,-0.208215898693784,-0.1502665699522203,0.002715796595951944,0.1469750365794204,0.1891088509683335,0.1060410342644908,-0.04298421215733216,-0.1581726565369832,-0.1655991733907199,-0.06433242730337559,0.07559119370385811,0.1612964670272104,0.138440446709381,0.02544713387345254,-0.1008298533389065,-0.1571614372635811,-0.1086060314457689,0.009999022660896667,0.1188226009426482,0.1466096370705062,0.07719750086663293,-0.04130102000267682,-0.1297111003612162,-0.1305715650764496,-0.04537649160839675,0.06778956329546182,0.1337499196955598,0.1100789444906126,0.01430634579621767,-0.08890920466593764,-0.1313503367336547,-0.08625142391580537])
# integrate(legendre_p_n(x),x,0,1)
int_pn = 8*np.pi**2*np.array([1,1/2,0,-1/8,0,1/16,0,-5/128,0,7/256,0,-21/1024,0,33/2048,0,-429/32768,0,715/65536,0,-2431/262144,0])
# integrate(legendre_p_n(x),x,-1,0)
#int_pn = [1,-1/2,0,1/8,0,-1/16,0,5/128,0,-7/256,0,21/1024,0,-33/2048,0,429/32768,0,-715/65536,0,2431/262144,0]
# integrate(legendre_p_n(x),x,-1,1/2)
#int_pn = 8*np.pi**2* np.array([1.5, -0.375, -0.1875, -0.0234375, 0.0585938, 0.0556641, 0.0102539, -0.0264587, -0.0288849, -0.00603104, 0.0157986, 0.018347, 0.00407732, -0.0107751, -0.0129594, -0.00298973, 0.00794653, 0.00977575, 0.00231217, -0.00617063, -0.00771173, -0.00185675, 0.0049702, 0.00628422, 0.00153339, -0.00411419, -0.00524864, -0.00129408, 0.00347853, 0.00446927, 0.00111112, -0.00299124, -0.00386539, -0.000967545, 0.00260799, 0.00338624, 0.000852429, -0.00230013, -0.00299852, 0.000758456, 0.0020484, 0.00267953, 0.000680563, -0.00183947, -0.00241335, -0.000615146, 0.00166379, 0.0021885, 0.000559576, -0.0015144, -0.00199652, -0.000511896])
# integrate(x * legendre_p_n(x),0,1)
#int_pn = [1/2,1/3,1/8,0,-1/48,0,1/128,0,-1/256,0,7/3072,0,-3/2048,0,33/32768,0,-143/196608,0,143/262144,0,-221/524288]
# integrate( (pi/2-arccos(x))^2 * legendre_p_n(x),0,1)
#int_pn = [0.4674011002723398,0.3668502750680843,0.2222222222222223,0.1010531421917553,0.0355555555555555,0.01484661888127219,0.01160997732426307,0.008973288625497218,0.005159989921895382,0.003068786426492212,0.002729250867771737]
# integrate( (pi-arccos(x)) * legendre_p_n(x),-1,1)
#int_pn =[3.141592653589793,0.7853981633974448,0,0.04908738521234036,0,0.01227184630308148,0,0.004793689962076684,0,0.002348908082027755,0,0.001321260789556922,0,0.815675979112751*10**-3,0,0.538468280884195*10**-3,0,0.373970709757952*10**-3,0,0.267526379439437*10**-3,0]
for n in range(N+1):
    disc._mu_hat[n,0,0] = int_pn[n]
disc._weights = np.real(disc._mu_hat[0,0,0])* np.ones([int(M),1],dtype=float) / int(M)
#points.local_coords
disc.f(points)
grad=disc.grad(points)
#grad.perform_geodesic_step(0)

vec = lorm.manif.TangentVectorArrayParameterized(points)
vec.coords = np.random.randn(M,9)
vec.coords = grad.coords
vec.coords *= 10
[f, q, s] =lorm.utils.eval_objective_function_with_quadratic_approximation(disc,vec,sample_size=100)
pl.plot(s,f,s,q)
pl.plot(s,(f-q)/s**3)
method = lorm.optim.SteepestDescentMethod(max_iter=20)
method = lorm.optim.ConjugateGradientMethod(max_iter=20)
for i in range(1):
    points = method.run(disc,points)
#
#
pl.plot(disc._eval_lengths(points.local_coords))
center = np.zeros(9)
for p in points.coords:
    center += p
center /= M
center = np.array([[-1,0.2,3],[0,1.4,8],[-1,0.3,0]])
center = so3.project_on_manifold(center.reshape(1,9))
center

quat_proj = np.zeros([M,3])

#center = so3.project_on_manifold(np.random.randn(1,9))

coords = so3.compute_parameterization(np.array([[0,np.pi/2-0.0001,np.pi/2],[np.pi,np.pi/2,3./2*np.pi],[0,np.pi/2,3./2*np.pi],[np.pi+0.00001,np.pi/2,np.pi/2]]))
tempc = coords
m2 = len(tempc)
coords = np.zeros([2*m2,9])
for i in range(m2):
    coords[2*i,:] = tempc[i,:]
for i in range(m2-1):
    coords[2*i+1,:] = so3.project_on_manifold(((tempc[i,:]+tempc[i+1,:])/2).reshape(1,9))
coords[2*m2-1,:] = so3.project_on_manifold(((.5*tempc[0,:]+.5*tempc[m2-1,:])).reshape(1,9))

m2 = len(coords)
M=m2
M

center = so3.compute_parameterization(np.array([[0,np.pi/4,0]]))
for i,p in enumerate(coords):
    coords[i,:] = (center.reshape(3,3).transpose()@p.reshape(3,3) @ center.reshape(3,3)).reshape(1,9)

#coords = so3.project_on_manifold(np.array([[0,0,-1],[-1,0,0],[0,1,0]]).reshape(1,9))
#so3.compute_inverse_parameterization(coords)
#coords


quat = np.zeros([M,4])
quat_proj = np.zeros([coords.shape[0],3])
#center = so3.compute_parameterization(np.array([[0,0,np.pi]]))
coords = points.coords
center = so3.compute_parameterization(np.array([[0,0,0*np.pi]]))#np.eye(3)
for i,p in enumerate(coords):
    matrix = center.reshape(3,3).transpose()@p.reshape(3,3)# @ center.reshape(3,3)
    quat[i,:] = lorm.manif.SO3.compute_quaternion_representation(matrix.reshape(9))
    quat_proj[i,:] = lorm.manif.Sphere3.compute_stereographicprojection(quat[i,:])
np.savetxt('line.txt',quat_proj)
