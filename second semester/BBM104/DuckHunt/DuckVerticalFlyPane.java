import javafx.scene.layout.Pane;

import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;

import javafx.util.Duration;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;

import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
/**
DuckVerticalFlyPane class extends Pane class and represents a pane that displays a vertical flying and falling duck
with animation and sound effects.
*/
// Hard part of this code was to make sure that duck hits the walls
public class DuckVerticalFlyPane extends Pane {
	private ImageView[] pngList = new ImageView[8];
	private String duckPath;
	 ImageView duck;
	private int duckNo;
	private int reverse = 1;
	private long x,y;
	private int dx;
	private int counter = 0;
	private Timeline animation;
	Timeline animation2;
	MediaPlayer mediaPlayer = new MediaPlayer(
			new Media(getClass().getResource(
					"assets/effects/DuckFalls.mp3").toExternalForm()));;
	
	public boolean isDead = false;
	/**
	 * Constructor for creating a new DuckVerticalFlyPane object.
	 * 
	 * @param duckPath the path of the duck image file
	 * @param lOrR true if the duck flies from left to right, false otherwise
	 * @param yCoord the initial y-coordinate of the duck
	 */
	DuckVerticalFlyPane(String duckPath,boolean lOrR,long yCoord){
        mediaPlayer.setVolume(DuckHunt.VOLUME);
        
		this.isDead = false;
		this.setPrefHeight(256*DuckHunt.SCALE);
		this.setPrefWidth(240*DuckHunt.SCALE);
		this.duckNo = 3;
		this.duckPath = "assets/"+duckPath;
		// create the image views for the duck and set their properties
		for (int i = 1; i < 9; i++) {
			String tmp = String.valueOf(i);
			Image tmp2 = new Image(this.duckPath+tmp+".png");
			pngList[i-1] = new ImageView(tmp2);
			pngList[i-1].setFitHeight(tmp2.getHeight()*DuckHunt.SCALE);
			pngList[i-1].setFitWidth(tmp2.getWidth()*DuckHunt.SCALE);
			//Animation for the fall 
			pngList[i-1].setMouseTransparent(true);

		}

		
		
		//Initial state of coordinates and amount of pixel change

		dx = (int)(DuckHunt.SCALE*256-this.pngList[4].getFitWidth())/10;
		if (lOrR) {
			this.x = 0;
			
		}else {
			this.x = Math.round(256*DuckHunt.SCALE-pngList[duckNo].getFitWidth());
			reverse = -reverse;
			dx = -dx;
		}
		this.y = yCoord;
		
		
		
		//left and right animation
		animation = new Timeline(
			    new KeyFrame(Duration.ZERO, e -> moveDuck()),

			    new KeyFrame(Duration.millis(280))
			);
	    animation.setCycleCount(Timeline.INDEFINITE);
	    animation.play(); // Start animation
		

		
		//Animation of Death

		
	    animation2 = new Timeline(
	    	    new KeyFrame(Duration.millis(0), e -> fallImage()),
	    	    new KeyFrame(Duration.millis(700), e -> fall())
	    	);

	    
		

		}
		// starts the fall animation.
		public void fall() {
			
			
			this.duck = pngList[7];
			this.getChildren().clear();
			
			duck.setLayoutX(x);
			duck.setLayoutY(y);

			duck.setScaleX(reverse);

			this.getChildren().add(duck);
			
			
			//fall animation
			animation = new Timeline(
				    new KeyFrame(Duration.ZERO, e -> fallDuck()),
				    new KeyFrame(Duration.millis(200))
				);
		    animation.setCycleCount(15);
		    animation.play(); // Start animation


	        
	        

		}
		public void fallDuck() {
			y+=35*DuckHunt.SCALE;
			duck.setLayoutX(x);
			duck.setLayoutY(y);
		}
		/**
		 * Changes the duck image from dlying duck to the falling duck 
		 */
		public void fallImage() {
		    mediaPlayer.play();
		    mediaPlayer.setOnEndOfMedia(() -> {
	            mediaPlayer.seek(Duration.ZERO);
	            mediaPlayer.stop();
		    });
			this.isDead = true;
			this.animation.stop();
			this.duck =pngList[6];
			this.getChildren().clear();
			
			duck.setLayoutX(x);
			duck.setLayoutY(y);
			duck.setScaleX(reverse);
			this.getChildren().add(duck);
		}
		public void updateDuckNo() {
			if (duckNo==5) {
				duckNo = 3;
			}else {
				duckNo++;
			}
		}
		public void updateDuck() {
			
			this.duck = pngList[duckNo];
			this.getChildren().clear();
			this.getChildren().add(duck);
		}

		protected void moveDuck() {

			
			updateDuckNo();
			updateDuck();
			
			
			if (counter == 10) {
				reverse = -reverse;
				counter = 0;
				dx = -dx;
			}else {
				counter++;
				x+=dx;
			}
			duck.setLayoutX(x);
			duck.setLayoutY(y);
			duck.setScaleX(reverse);

		}
	

}
