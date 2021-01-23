from flask import Flask
from tasks import make_celery

flask_app = Flask(__name__)
flask_app.config.update(
	CELERY_BROKER_URL='redis://localhost:6379',
	CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(flask_app)

@celery.task()
def add_together(a, b):
	sums = a + b
	print(sums)
	return sums

if __name__ == '__main__':
	app.secret_key = 'super secret key'
	app.config['SESSION_TYPE'] = 'filesystem'
	sess.init_app(app)

	app.run(debug = True)

