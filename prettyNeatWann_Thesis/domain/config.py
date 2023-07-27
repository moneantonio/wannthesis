from collections import namedtuple
import numpy as np
#from wann_train import the_game

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels','ram','full'])

games = {}
the_game = 'Pong'
#next_game = str((np.random.choice(atari_env_names,1)[0]))

# -- Atari512 - Phoenix ------------------------------------------------------- --#

atari_stack1024 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 1024,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(1024,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent129','latent130','latent131','latent132','latent133',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent129','latent130','latent131','latent132','latent133',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack1024'] = atari_stack1024
# -- Atari512 - Phoenix ------------------------------------------------------- --#

atari_stack512 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 512,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(512,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent129','latent130','latent131','latent132','latent133',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack512'] = atari_stack512

# -- Atari256 - Phoenix ------------------------------------------------------- --#

atari_stack256 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 256,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(256,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent129','latent130','latent131','latent132','latent133',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack256'] = atari_stack256

# -- Atari192 - Phoenix ------------------------------------------------------- --#

atari_stack192 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 192,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(192,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','latent129','latent130','latent131','latent132','latent133',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack192'] = atari_stack192

# -- Atari128 - VAE ------------------------------------------------------- --#

atari_stack128_vae = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 128,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(128,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire'],
  ram=1
  ,full=False
  )
games['atari_stack128_vae'] = atari_stack128_vae

# -- Atari128 - RAM ------------------------------------------------------- --#

atari_stack128_ram = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 128,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(128,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 5000,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire'],
  ram=0
  ,full=False
  )
games['atari_stack128_ram'] = atari_stack128_ram

# -- Atari128 - RAW ------------------------------------------------------- --#

atari_stack_raw= Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 50*50*3,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=0,#np.full(8,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = [['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','latent65',\
                   'latent66','latent67','latent68','latent69','latent70',\
                   'latent71','latent72','latent73','latent74','latent75',\
                   'latent76','latent77','latent78','latent79','latent80',\
                   'latent81','latent82','latent83','latent84','latent85',\
                   'latent86','latent87','latent88','latent89','latent90',\
                   'latent91','latent92','latent93','latent94','latent95',\
                   'latent96','latent97','latent98','latent99','latent100',\
                   'latent101','latent102','latent103','latent104','latent105',\
                   'latent106','latent107','latent108','latent109','latent110',\
                   'latent111','latent112','latent113','latent114','latent115',\
                   'latent116','latent117','latent18','latent119','latent120',\
                   'latent121','latent122','latent123','latent124','latent125', 
                   'latent126','latent127','latent128']*166,'noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire'],
  ram=2
  ,full=False
  )
games['atari_stack_raw'] = atari_stack_raw

# -- Atari64 - Phoenix ------------------------------------------------------- --#

atari_stack64 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 64,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(64,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','latent33','latent34','latent35',\
                   'latent36','latent37','latent38','latent39','latent40',\
                   'latent41','latent42','latent43','latent44','latent45',\
                   'latent46','latent47','latent48','latent49','latent50',\
                   'latent51','latent52','latent53','latent54','latent55',\
                   'latent56','latent57','latent58','latent59','latent60',\
                   'latent61','latent62','latent63','latent64','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack64'] = atari_stack64

# -- Atari32 - Phoenix ------------------------------------------------------- --#

atari_stack32 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 32,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(32,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack32'] = atari_stack32

# -- Atari16 - Phoenix ------------------------------------------------------- --#

atari_stack16 = Game(env_name=the_game,
  actionSelect='softmax',
  input_size = 16,
  output_size = 18,
  time_factor = 0,
  layers = None,
  i_act=np.full(16,1),
  h_act=[1,2,3,4,5,6,7,8,9,10,11],
  o_act=np.full(18,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 1500,
  output_noise=[False, False, False,False, False, False,False, False, False,False, False, False,False, False, False,False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','noop','fire','up','right',\
                   'left','down','upright','upleft','downright','downleft',\
                   'upfire','rightfire','leftfire','downfire','uprightfire','upleftfire','downrightfire','downleftfire']
  ,ram=1
  ,full=False
  )
games['atari_stack16'] = atari_stack16

# -- Car Racing  --------------------------------------------------------- -- #

# > 32 latent vectors (includes past frames)
vae_racing_stack = Game(env_name='VAERacingStack-v0',
  actionSelect='all', # all, soft, hard
  input_size=32,
  output_size=3,
  time_factor=0,
  layers=[10, 0],
  i_act=np.full(32,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(3,1),
  weightCap = 2.0,
  noise_bias=0.0,
  max_episode_length = 500,
  output_noise=[False, False, False],
  in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                   'latent06','latent07','latent08','latent09','latent10',\
                   'latent11','latent12','latent13','latent14','latent15',\
                   'latent16','latent17','latent18','latent19','latent20',\
                   'latent21','latent22','latent23','latent24','latent25',\
                   'latent26','latent27','latent28','latent29','latent30',\
                   'latent31','latent32','steer'   ,'gas'     ,'brakes']
  ,ram=1
  ,full=False
)
games['vae_racing_stack'] = vae_racing_stack

# > 16 latent vectors (current frame only)
vae_racing = vae_racing_stack._replace(\
  env_name='VAERacing-v0', input_size=16, i_act=np.full(16,1),\
    in_out_labels = ['latent01','latent02','latent03','latent04','latent05',\
                     'latent06','latent07','latent08','latent09','latent10',\
                     'latent11','latent12','latent13','latent14','latent15',\
                     'latent16','steer'   ,'gas'     ,'brakes']  )
games['vae_racing'] = vae_racing


# -- Digit Classification ------------------------------------------------ -- #

# > Scikit learn digits data set
classify = Game(env_name='Classify_digits',
  actionSelect='softmax', # all, soft, hard
  input_size=64,
  output_size=10,
  time_factor=0,
  layers=[128,9],
  i_act=np.full(64,1),
  h_act=[1,3,4,5,6,7,8,9,10], # No step function
  o_act=np.full(10,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 0,
  in_out_labels = [],ram=1
  ,full=False
)
L = [list(range(1, classify.input_size)),\
     list(range(0, classify.output_size))]
label = [item for sublist in L for item in sublist]
classify = classify._replace(in_out_labels=label)
games['digits'] = classify


# > MNIST [28x28] data set
mnist784 = classify._replace(\
  env_name='Classify_mnist784', input_size=784, i_act=np.full(784,1))
L = [list(range(1, mnist784.input_size)),\
     list(range(0, mnist784.output_size))]
label = [item for sublist in L for item in sublist]
mnist784 = mnist784._replace(in_out_labels=label)
games['mnist784'] = mnist784

# > MNIST [16x16] data set
mnist256 = classify._replace(\
  env_name='Classify_mnist256', input_size=256, i_act=np.full(256,1))
L = [list(range(1, mnist256.input_size)),\
     list(range(0, mnist256.output_size))]
label = [item for sublist in L for item in sublist]
mnist256 = mnist256._replace(in_out_labels=label)
games['mnist256'] = mnist256


# -- Cart-pole Swingup --------------------------------------------------- -- #

# > Slower reaction speed
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
  actionSelect='all', # all, soft, hard
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[5, 5],
  i_act=np.full(5,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(1,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 200,
  in_out_labels = ['x','x_dot','cos(theta)','sin(theta)','theta_dot',
                   'force'],ram=1
  ,full=False
)
games['swingup_hard'] = cartpole_swingup

# > Normal reaction speed
cartpole_swingup = cartpole_swingup._replace(\
    env_name='CartPoleSwingUp', max_episode_length=1000)
games['swingup'] = cartpole_swingup


# -- Bipedal Walker ------------------------------------------------------ -- #

# > Flat terrain
biped = Game(env_name='BipedalWalker-v2',
  actionSelect='all', # all, soft, hard
  input_size=24,
  output_size=4,
  time_factor=0,
  layers=[40, 40],
  i_act=np.full(24,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(4,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 400,
  in_out_labels = [
  'hull_angle','hull_vel_angle','vel_x','vel_y',
  'hip1_angle','hip1_speed','knee1_angle','knee1_speed','leg1_contact',
  'hip2_angle','hip2_speed','knee2_angle','knee2_speed','leg2_contact',
  'lidar_0','lidar_1','lidar_2','lidar_3','lidar_4',
  'lidar_5','lidar_6','lidar_7','lidar_8','lidar_9',
  'hip_1','knee_1','hip_2','knee_2'],ram=1,full=False
)
games['biped'] = biped

# > Hilly Terrain
bipedmed = biped._replace(env_name='BipedalWalkerMedium-v2')
games['bipedmedium'] = bipedmed

# > Obstacles, hills, and pits
bipedhard = biped._replace(env_name='BipedalWalkerHardcore-v2')
games['bipedhard'] = bipedhard


# -- Bullet -------------------------------------------------------------- -- #

# > Quadruped ant
bullet_ant = Game(env_name='AntBulletEnv-v0',
  actionSelect='all', # all, soft, hard
  input_size=28,
  output_size=8,
  layers=[64, 32],
  time_factor=1000,
  i_act=np.full(28,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(8,1),
  weightCap = 2.0,
  noise_bias=0.0,
  output_noise=[False, False, True],
  max_episode_length = 1000,
  in_out_labels = [],ram=1,full=False
)
games['bullet_ant'] = bullet_ant